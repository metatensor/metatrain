from __future__ import annotations

import copy
import importlib
import math
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System


if TYPE_CHECKING:
    from types import ModuleType

    from pyscf.gto import Mole


RIAuxBasis = str | dict[str, str]


# ── PySCF imports ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _import_pyscf_modules() -> tuple[ModuleType, ModuleType]:
    try:
        gto = importlib.import_module("pyscf.gto")
        elements = importlib.import_module("pyscf.data.elements")
    except ModuleNotFoundError as err:
        raise ImportError(
            "RI overlap losses require `pyscf` to compute auxiliary overlap matrices."
        ) from err

    return gto, elements


def _build_etb_basis_for_element(ref_basis: str, symbol: str, ratio: float) -> list:
    """Build an uncontracted ETB basis for one element from a reference basis.

    Parses ``ref_basis`` (a named PySCF basis) to discover which angular-momentum
    channels are present and the exponent range for each.  For every channel ``l``
    the function constructs an even-tempered sequence of ``n`` uncontracted
    primitives starting at the minimum exponent of the reference and spaced by
    ``ratio``, choosing ``n`` to cover at least the full exponent range of the
    reference.  The result is returned in the format accepted by ``gto.etbs`` /
    ``mol.basis``.

    :param ref_basis: Name of the PySCF reference basis (e.g. ``"def2-svp"``).
    :param symbol: Element symbol (e.g. ``"O"``).
    :param ratio: Even-tempering ratio β (consecutive exponents differ by this factor).
    :return: Basis specification list suitable for ``mol.basis[symbol]``.
    """
    gto, _ = _import_pyscf_modules()
    ref = gto.basis.load(ref_basis, symbol)

    # Collect all primitive exponents per angular momentum from the reference.
    l_exps: dict[int, list[float]] = {}
    for shell in ref:
        l = int(shell[0])
        for prim in shell[1:]:
            if isinstance(prim, (list, tuple)) and prim:
                l_exps.setdefault(l, []).append(float(prim[0]))

    etb_specs = []
    for l in sorted(l_exps):
        exps = sorted(l_exps[l])
        alpha_min = exps[0]
        alpha_max = exps[-1]
        if len(exps) > 1 and alpha_max > alpha_min:
            # Enough primitives to cover [alpha_min, alpha_max] with spacing ratio.
            n = max(
                len(exps),
                round(math.log(alpha_max / alpha_min) / math.log(ratio)) + 1,
            )
        else:
            n = len(exps)
        etb_specs.append((l, n, alpha_min, ratio))

    return gto.etbs(etb_specs)


@lru_cache(maxsize=None)
def _load_auxiliary_basis(
    aux_basis: str, atomic_numbers: tuple[int, ...]
) -> dict[str, object]:
    """Load and cache parsed auxiliary basis data for the requested elements.

    Supports two formats for ``aux_basis``:

    - A plain PySCF basis name (e.g. ``"def2-universal-jfit"``), loaded via
      ``gto.basis.load``.
    - An even-tempered basis specification ``"etb:<ref_basis>:<ratio>"``
      (e.g. ``"etb:def2-svp:2.0"``), which derives an uncontracted ETB from
      ``<ref_basis>`` using the given ratio for each element.
    """
    gto, elements = _import_pyscf_modules()

    etb_parts = aux_basis.split(":")
    is_etb = len(etb_parts) == 3 and etb_parts[0].lower() == "etb"

    basis: dict[str, object] = {}
    for atomic_number in atomic_numbers:
        symbol = elements.ELEMENTS[atomic_number]
        if is_etb:
            basis[symbol] = _build_etb_basis_for_element(
                etb_parts[1], symbol, float(etb_parts[2])
            )
        else:
            basis[symbol] = gto.basis.load(aux_basis, symbol)

    return basis


# ── extra_data key helpers ─────────────────────────────────────────────────────

def overlap_matrix_name(target_name: str) -> str:
    """Return the ``extra_data`` key used for a target's overlap matrix."""
    return f"{target_name}_overlap_matrix"


def coulomb_matrix_name(target_name: str) -> str:
    """Return the ``extra_data`` key used for a target's Coulomb matrix."""
    return f"{target_name}_coulomb_matrix"


def metric_matrix_name(target_name: str, metric: str) -> str:
    """Return the ``extra_data`` key for a target's two-centre metric matrix.

    :param target_name: Name of the RI-coefficient target.
    :param metric: ``"overlap"`` (S) or ``"coulomb"`` (J).
    """
    if metric == "overlap":
        return overlap_matrix_name(target_name)
    elif metric == "coulomb":
        return coulomb_matrix_name(target_name)
    else:
        raise ValueError(
            f"Unknown RI metric '{metric}'. Supported values are 'overlap' and 'coulomb'."
        )


def ri_projections_name(target_name: str) -> str:
    """Return the ``extra_data`` key for a target's RI projections w = S c_RI."""
    return f"{target_name}_projections"


def ri_density_fit_constant_name(target_name: str) -> str:
    """Return the ``extra_data`` key for the pre-computed c_RI^T w_RI constant."""
    return f"{target_name}_density_fit_constant"


def resolve_ri_aux_basis(target_name: str, ri_aux_basis: RIAuxBasis) -> str:
    """Resolve the auxiliary basis configured for a given RI target."""

    if isinstance(ri_aux_basis, str):
        return ri_aux_basis

    if target_name in ri_aux_basis:
        return ri_aux_basis[target_name]

    available_targets = ", ".join(sorted(ri_aux_basis))
    raise ValueError(
        f"No RI auxiliary basis configured for target '{target_name}'. "
        f"Available targets: {available_targets}."
    )


# ── Molecule / integral construction ──────────────────────────────────────────

def _system_to_atom_string(system: System) -> str:
    _, elements = _import_pyscf_modules()

    atomic_numbers = system.types.detach().cpu().tolist()
    positions = system.positions.detach().cpu().tolist()

    atoms = []
    for atomic_number, (x, y, z) in zip(atomic_numbers, positions, strict=True):
        symbol = elements.ELEMENTS[int(atomic_number)]
        atoms.append(f"{symbol}  {x:.12f}  {y:.12f}  {z:.12f}")

    return "\n".join(atoms)


def build_auxiliary_molecule(system: System, aux_basis: str) -> Mole:
    """
    Build a PySCF molecule carrying the auxiliary basis used for RI coefficients.

    The molecule is only used for integral evaluation.

    :param system: Atomistic system with positions in Angstrom.
    :param aux_basis: PySCF auxiliary basis name.
    :return: Built PySCF molecule in spherical-harmonic form.
    """
    gto, _ = _import_pyscf_modules()
    atomic_numbers = tuple(
        sorted({int(n) for n in system.types.detach().cpu().tolist()})
    )

    mol = gto.Mole()
    mol.atom = _system_to_atom_string(system)
    mol.basis = copy.deepcopy(_load_auxiliary_basis(aux_basis, atomic_numbers))
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.spin = None
    mol.cart = False
    mol.build()
    return mol


def compute_overlap_matrix(system: System, aux_basis: str) -> torch.Tensor:
    """
    Compute the two-centre overlap matrix for one system and auxiliary basis.

    :param system: Atomistic system with positions in Angstrom.
    :param aux_basis: PySCF auxiliary basis name.
    :return: Dense overlap matrix ``S`` in PySCF AO order, float64.
    """
    auxmol = build_auxiliary_molecule(system, aux_basis)
    return torch.from_numpy(auxmol.intor("int1e_ovlp")).to(torch.float64)


def compute_coulomb_matrix(system: System, aux_basis: str) -> torch.Tensor:
    """
    Compute the two-centre Coulomb (ERI) matrix for one system and auxiliary basis.

    :param system: Atomistic system with positions in Angstrom.
    :param aux_basis: PySCF auxiliary basis name.
    :return: Dense Coulomb matrix ``J`` in PySCF AO order, float64.
    """
    auxmol = build_auxiliary_molecule(system, aux_basis)
    return torch.from_numpy(auxmol.intor("int2c2e")).to(torch.float64)


def compute_metric_matrix(system: System, aux_basis: str, metric: str) -> torch.Tensor:
    """
    Compute the two-centre metric matrix (overlap or Coulomb) for a system.

    :param system: Atomistic system with positions in Angstrom.
    :param aux_basis: PySCF auxiliary basis name.
    :param metric: ``"overlap"`` for S or ``"coulomb"`` for J.
    :return: Dense metric matrix in PySCF AO order, float64.
    """
    if metric == "overlap":
        return compute_overlap_matrix(system, aux_basis)
    elif metric == "coulomb":
        return compute_coulomb_matrix(system, aux_basis)
    else:
        raise ValueError(
            f"Unknown RI metric '{metric}'. Supported values are 'overlap' and 'coulomb'."
        )


# ── TensorMap packing / unpacking ──────────────────────────────────────────────

def pack_two_center_matrices(matrices: list[torch.Tensor]) -> TensorMap:
    """
    Pack variable-size per-system two-centre matrices into one padded TensorMap.

    :param matrices: Dense square matrices, one per system.
    :return: Single-block TensorMap with samples ``(system, basis_size)``.
    """
    if len(matrices) == 0:
        raise ValueError("Expected at least one two-centre matrix to pack.")

    max_basis = max(matrix.shape[0] for matrix in matrices)
    dtype = matrices[0].dtype
    device = matrices[0].device
    basis_sizes = torch.tensor(
        [[matrix.shape[0]] for matrix in matrices], dtype=torch.int32, device=device
    )

    values = torch.full(
        (len(matrices), max_basis, max_basis),
        fill_value=torch.nan,
        dtype=dtype,
        device=device,
    )
    for i_system, matrix in enumerate(matrices):
        n_basis = matrix.shape[0]
        values[i_system, :n_basis, :n_basis] = matrix

    block = TensorBlock(
        values=values,
        samples=Labels(
            names=["system", "basis_size"],
            values=torch.hstack(
                [
                    torch.arange(
                        len(matrices), dtype=torch.int32, device=device
                    ).reshape(-1, 1),
                    basis_sizes,
                ]
            ),
        ),
        components=[
            Labels(
                names=["basis_1"],
                values=torch.arange(
                    max_basis, dtype=torch.int32, device=device
                ).reshape(-1, 1),
            )
        ],
        properties=Labels(
            names=["basis_2"],
            values=torch.arange(max_basis, dtype=torch.int32, device=device).reshape(
                -1, 1
            ),
        ),
    )
    return TensorMap(Labels.single().to(device=device), [block])


def packed_two_center_basis_sizes(batched_matrices: TensorMap) -> torch.Tensor:
    """Return the stored per-system basis sizes for packed two-centre matrices."""
    return batched_matrices.block().samples.values[:, 1].to(torch.int64)


def unpack_two_center_matrices(
    batched_matrices: TensorMap, basis_sizes: list[int]
) -> list[torch.Tensor]:
    """
    Unpack a padded batch of two-centre matrices back to dense per-system matrices.

    :param batched_matrices: Padded batched two-centre matrices.
    :param basis_sizes: Physical basis size for each system.
    :return: Dense matrices cropped to each system size.
    """
    block = batched_matrices.block()
    if len(block.samples) != len(basis_sizes):
        raise ValueError(
            "The number of packed two-centre matrices does not match the basis sizes."
        )

    packed_basis_sizes = packed_two_center_basis_sizes(batched_matrices).tolist()
    for i_system, n_basis in enumerate(basis_sizes):
        actual = packed_basis_sizes[i_system]
        if n_basis > actual:
            raise ValueError(
                "At least one requested two-centre matrix basis size exceeds the "
                "packed matrix size."
            )
        if n_basis != actual:
            raise ValueError(
                "The RI target size does not match the configured auxiliary basis. "
                "Check that 'ri_aux_basis' matches the RI coefficients stored in the "
                "dataset."
            )

    return [
        block.values[i_system, :n_basis, :n_basis]
        for i_system, n_basis in enumerate(basis_sizes)
    ]


def compute_batched_overlap_matrices(
    systems: list[System], aux_basis: str
) -> TensorMap:
    """
    Compute padded per-system overlap matrices for a batch of systems.

    :param systems: Systems in one batch.
    :param aux_basis: PySCF auxiliary basis name.
    :return: Padded batch of overlap matrices.
    """
    matrices = [compute_overlap_matrix(system, aux_basis) for system in systems]
    return pack_two_center_matrices(matrices)


def compute_batched_coulomb_matrices(
    systems: list[System], aux_basis: str
) -> TensorMap:
    """
    Compute padded per-system Coulomb matrices for a batch of systems.

    :param systems: Systems in one batch.
    :param aux_basis: PySCF auxiliary basis name.
    :return: Padded batch of Coulomb matrices.
    """
    matrices = [compute_coulomb_matrix(system, aux_basis) for system in systems]
    return pack_two_center_matrices(matrices)


def compute_batched_metric_matrices(
    systems: list[System], aux_basis: str, metric: str
) -> TensorMap:
    """
    Compute padded per-system metric matrices (overlap or Coulomb) for a batch.

    :param systems: Systems in one batch.
    :param aux_basis: PySCF auxiliary basis name.
    :param metric: ``"overlap"`` or ``"coulomb"``.
    :return: Padded batch of metric matrices.
    """
    matrices = [compute_metric_matrix(system, aux_basis, metric) for system in systems]
    return pack_two_center_matrices(matrices)


# ── Collate transforms ────────────────────────────────────────────────────────

def get_overlap_matrices_transform(
    target_to_aux_basis: Mapping[str, str],
) -> Callable:
    """Create a collate transform that attaches per-target overlap matrices."""

    def transform(
        systems: list[System],
        targets: dict[str, TensorMap],
        extra: dict[str, TensorMap],
    ) -> tuple[list[System], dict[str, TensorMap], dict[str, TensorMap]]:
        cache: dict[str, TensorMap] = {}
        for target_name, aux_basis in target_to_aux_basis.items():
            if aux_basis not in cache:
                cache[aux_basis] = compute_batched_overlap_matrices(systems, aux_basis)
            extra[overlap_matrix_name(target_name)] = cache[aux_basis]
        return systems, targets, extra

    return transform


def get_coulomb_matrices_transform(
    target_to_aux_basis: Mapping[str, str],
) -> Callable:
    """Create a collate transform that attaches per-target Coulomb matrices."""

    def transform(
        systems: list[System],
        targets: dict[str, TensorMap],
        extra: dict[str, TensorMap],
    ) -> tuple[list[System], dict[str, TensorMap], dict[str, TensorMap]]:
        cache: dict[str, TensorMap] = {}
        for target_name, aux_basis in target_to_aux_basis.items():
            if aux_basis not in cache:
                cache[aux_basis] = compute_batched_coulomb_matrices(systems, aux_basis)
            extra[coulomb_matrix_name(target_name)] = cache[aux_basis]
        return systems, targets, extra

    return transform


def get_metric_matrices_transform(
    target_to_aux_basis: Mapping[str, str],
    metric: str,
) -> Callable:
    """Create a collate transform that attaches per-target metric matrices.

    :param target_to_aux_basis: Mapping from target name to PySCF auxiliary basis name.
    :param metric: ``"overlap"`` or ``"coulomb"``.
    """
    if metric == "overlap":
        return get_overlap_matrices_transform(target_to_aux_basis)
    elif metric == "coulomb":
        return get_coulomb_matrices_transform(target_to_aux_basis)
    else:
        raise ValueError(
            f"Unknown RI metric '{metric}'. Supported values are 'overlap' and 'coulomb'."
        )


def get_density_fit_constant_transform(
    target_to_projections_key: Mapping[str, str],
) -> Callable:
    """
    Create a collate transform that pre-computes the per-system density-fit constant.

    For each target, computes ``c_RI^T w_RI`` (= ``c_RI^T S c_RI`` when
    ``w_RI = S c_RI``) and stores the result in ``extra_data`` as a scalar
    TensorMap.  This constant bounds the density-fit loss from below at zero and
    has no gradient w.r.t. model parameters.

    **This transform must run before the CM-removal and scale-removal transforms**
    so that ``targets`` still contains raw RI coefficients in physical units.

    :param target_to_projections_key: mapping from RI-coefficient target name to the
        ``extra_data`` key under which the corresponding projections ``w = M c_RI``
        are stored.
    """

    def transform(
        systems: list[System],
        targets: dict[str, TensorMap],
        extra: dict[str, TensorMap],
    ) -> tuple[list[System], dict[str, TensorMap], dict[str, TensorMap]]:
        for target_name, proj_key in target_to_projections_key.items():
            if target_name not in targets or proj_key not in extra:
                continue

            c_map = targets[target_name]
            w_map = extra[proj_key]

            first_block = c_map.block(c_map.keys[0])
            sys_idx_col = first_block.samples.values[:, 0]
            n_systems = int(sys_idx_col.max().item()) + 1
            device = first_block.values.device
            dtype = first_block.values.dtype

            constants = torch.zeros(n_systems, device=device, dtype=dtype)

            for key in c_map.keys:
                c_block = c_map.block(key)
                w_block = w_map.block(key)
                sys_idx = c_block.samples.values[:, 0].long()

                c_vals = c_block.values  # [n_samples, n_m, n_radial]
                w_vals = w_block.values

                # NaN entries are padding; zero them out before summing.
                mask = ~(torch.isnan(c_vals) | torch.isnan(w_vals))
                contrib = torch.where(mask, c_vals * w_vals, torch.zeros_like(c_vals))
                per_sample = contrib.sum(dim=[1, 2])  # [n_samples]
                constants.scatter_add_(0, sys_idx, per_sample)

            # Pack as a scalar TensorMap: one value per system.
            const_block = TensorBlock(
                values=constants.unsqueeze(-1),  # [n_systems, 1]
                samples=Labels(
                    names=["system"],
                    values=torch.arange(
                        n_systems, dtype=torch.int32, device=device
                    ).reshape(-1, 1),
                ),
                components=[],
                properties=Labels(
                    names=["_"],
                    values=torch.zeros((1, 1), dtype=torch.int32, device=device),
                ),
            )
            extra[ri_density_fit_constant_name(target_name)] = TensorMap(
                Labels.single().to(device=device), [const_block]
            )

        return systems, targets, extra

    return transform
