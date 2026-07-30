"""Auxiliary-basis two-centre metrics for density (RI-coefficient) losses.

A scalar field expanded on an atom-centred auxiliary basis,
:math:`\\rho(r) = \\sum_i c_i \\phi_i(r)`, has errors that are quadratic forms in the
coefficient residual, weighted by a two-centre metric :math:`M`:

* **overlap**, :math:`S_{ij} = \\int \\phi_i \\phi_j \\, dr`, giving the real-space
  L2 error :math:`\\int |\\Delta\\rho(r)|^2 dr`;
* **Coulomb**, :math:`J_{ij} = \\iint \\phi_i(r) |r - r'|^{-1} \\phi_j(r')`,
  giving the electrostatic self-energy of the residual.

Neither is preferred a priori: they weight different length scales of the error, and
which one trains better is an empirical question.

This module computes those metrics with PySCF and packs them for the dataloader to
carry to the density losses in :py:mod:`metatrain.utils.loss`.

PySCF is an optional dependency, imported lazily, so metatrain works without it
unless a density loss is actually configured.

**Positions are read in Angstrom**, matching the ``length_unit`` that metatrain
datasets declare; the collate transforms run before any unit conversion.

**Non-periodic systems only.** The metric is built from a molecular PySCF ``Mole``,
which ignores the cell; periodic systems are rejected rather than silently scored
against a molecular metric.
"""

from __future__ import annotations

import copy
import functools
import importlib
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System


if TYPE_CHECKING:
    from types import ModuleType

    from pyscf.gto import Mole


METRICS = ("overlap", "coulomb")


@lru_cache(maxsize=1)
def _import_pyscf() -> Tuple["ModuleType", "ModuleType"]:
    """Import PySCF lazily, with an actionable error if it is missing.

    :return: The ``pyscf.gto`` and ``pyscf.data.elements`` modules.
    """
    try:
        gto = importlib.import_module("pyscf.gto")
        elements = importlib.import_module("pyscf.data.elements")
    except ModuleNotFoundError as err:
        raise ImportError(
            "density losses require `pyscf` to compute auxiliary-basis metric "
            "matrices; install it with `pip install pyscf`."
        ) from err
    return gto, elements


def _build_etb_basis(
    ao_basis: str, atomic_numbers: Tuple[int, ...], beta: float
) -> Dict[str, object]:
    """Build an even-tempered auxiliary basis with ``pyscf.df.aug_etb``.

    Constructs a dummy molecule holding one atom of each requested element in the
    *orbital* basis ``ao_basis``, then calls ``aug_etb``. This reproduces how
    SCFBench-style RI datasets generate their auxiliary basis, so that the metric
    matches the basis the reference coefficients were fitted in.

    :param ao_basis: Orbital (not auxiliary) basis name, e.g. ``"def2-svp"``.
    :param atomic_numbers: Unique atomic numbers to build the basis for.
    :param beta: Even-tempering ratio.
    :return: Mapping from element symbol to basis specification.
    """
    gto, elements = _import_pyscf()
    df = importlib.import_module("pyscf.df")

    symbols = [elements.ELEMENTS[n] for n in atomic_numbers]
    # Spread the atoms out so the dummy molecule builds without warnings.
    mol = gto.Mole()
    mol.atom = "\n".join(f"{s} 0.0 0.0 {i * 10.0}" for i, s in enumerate(symbols))
    mol.basis = ao_basis
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.spin = None
    mol.build()

    return df.aug_etb(mol, beta=beta)


@lru_cache(maxsize=None)
def _load_auxiliary_basis(
    aux_basis: str, atomic_numbers: Tuple[int, ...]
) -> Dict[str, object]:
    """Load (and cache) the auxiliary basis for a set of elements.

    Two forms of ``aux_basis`` are supported:

    * a PySCF basis name, e.g. ``"def2-universal-jfit"``;
    * an even-tempered specification ``"etb:<ao_basis>:<beta>"``, e.g.
      ``"etb:def2-svp:2.0"``, which is built via :func:`_build_etb_basis`.

    :param aux_basis: The auxiliary basis, in either form.
    :param atomic_numbers: Unique atomic numbers to load the basis for.
    :return: Mapping from element symbol to basis specification.
    """
    gto, elements = _import_pyscf()

    parts = aux_basis.split(":")
    if len(parts) == 3 and parts[0].lower() == "etb":
        return _build_etb_basis(parts[1], atomic_numbers, float(parts[2]))

    return {
        elements.ELEMENTS[n]: gto.basis.load(aux_basis, elements.ELEMENTS[n])
        for n in atomic_numbers
    }


# ── extra_data key helpers ────────────────────────────────────────────────────


def overlap_matrix_name(target_name: str) -> str:
    """Return the ``extra_data`` key holding a target's overlap matrices.

    :param target_name: Name of the RI-coefficient target.
    :return: The ``extra_data`` key.
    """
    return f"{target_name}_overlap_matrix"


def coulomb_matrix_name(target_name: str) -> str:
    """Return the ``extra_data`` key holding a target's Coulomb matrices.

    :param target_name: Name of the RI-coefficient target.
    :return: The ``extra_data`` key.
    """
    return f"{target_name}_coulomb_matrix"


def metric_matrix_name(target_name: str, metric: str) -> str:
    """Return the ``extra_data`` key for a target's two-centre metric matrices.

    :param target_name: Name of the RI-coefficient target.
    :param metric: ``"overlap"`` or ``"coulomb"``.
    :return: The ``extra_data`` key.
    """
    if metric == "overlap":
        return overlap_matrix_name(target_name)
    if metric == "coulomb":
        return coulomb_matrix_name(target_name)
    raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}.")


def ri_projections_name(target_name: str) -> str:
    """Return the ``extra_data`` key for a target's projections ``w = M c_ref``.

    :param target_name: Name of the RI-coefficient target.
    :return: The ``extra_data`` key.
    """
    return f"{target_name}_projections"


def ri_density_fit_constant_name(target_name: str) -> str:
    """Return the ``extra_data`` key for the pre-computed ``c_ref^T w`` constant.

    :param target_name: Name of the RI-coefficient target.
    :return: The ``extra_data`` key.
    """
    return f"{target_name}_density_fit_constant"


# ── Molecule / integral construction ──────────────────────────────────────────


def build_auxiliary_molecule(system: System, aux_basis: str) -> "Mole":
    """
    Build a PySCF molecule carrying the auxiliary basis, for integral evaluation.

    :param system: System whose positions are interpreted as Angstrom.
    :param aux_basis: Auxiliary basis name or ``"etb:<ao_basis>:<beta>"``.
    :return: A built molecule in spherical-harmonic (not Cartesian) form, which is
        the convention the RI coefficients follow.
    """
    gto, elements = _import_pyscf()

    if bool(system.pbc.any()):
        raise NotImplementedError(
            "density losses are implemented for non-periodic systems only: the "
            "auxiliary-basis metric is built from a molecular PySCF `Mole`, which "
            "ignores the cell, so a periodic system would silently be scored against "
            "a molecular metric. Supporting periodicity needs lattice sums (and for "
            "the Coulomb metric an Ewald treatment, since the sum is only "
            "conditionally convergent), not just passing the cell through."
        )

    types = system.types.detach().cpu().tolist()
    positions = system.positions.detach().cpu().tolist()
    atomic_numbers = tuple(sorted({int(t) for t in types}))

    mol = gto.Mole()
    mol.atom = "\n".join(
        f"{elements.ELEMENTS[int(t)]}  {x:.12f}  {y:.12f}  {z:.12f}"
        for t, (x, y, z) in zip(types, positions, strict=True)
    )
    # ``_load_auxiliary_basis`` is cached, and ``Mole.build`` mutates its basis.
    mol.basis = copy.deepcopy(_load_auxiliary_basis(aux_basis, atomic_numbers))
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.spin = None
    mol.cart = False
    mol.build()
    return mol


def compute_metric_matrix(system: System, aux_basis: str, metric: str) -> torch.Tensor:
    """
    Compute a two-centre metric matrix of the auxiliary basis for one system.

    :param system: System whose positions are interpreted as Angstrom.
    :param aux_basis: Auxiliary basis name or ``"etb:<ao_basis>:<beta>"``.
    :param metric: ``"overlap"`` for S (``int1e_ovlp``) or ``"coulomb"`` for J
        (``int2c2e``).
    :return: Dense ``(n_basis, n_basis)`` matrix in PySCF AO order, float64.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}.")
    integral = "int1e_ovlp" if metric == "overlap" else "int2c2e"
    auxmol = build_auxiliary_molecule(system, aux_basis)
    return torch.from_numpy(auxmol.intor(integral)).to(torch.float64)


def compute_overlap_matrix(system: System, aux_basis: str) -> torch.Tensor:
    """Two-centre overlap matrix ``S``; see :func:`compute_metric_matrix`.

    :param system: System whose positions are interpreted as Angstrom.
    :param aux_basis: Auxiliary basis name or ``"etb:<ao_basis>:<beta>"``.
    :return: Dense ``(n_basis, n_basis)`` matrix in PySCF AO order, float64.
    """
    return compute_metric_matrix(system, aux_basis, "overlap")


def compute_coulomb_matrix(system: System, aux_basis: str) -> torch.Tensor:
    """Two-centre Coulomb matrix ``J``; see :func:`compute_metric_matrix`.

    :param system: System whose positions are interpreted as Angstrom.
    :param aux_basis: Auxiliary basis name or ``"etb:<ao_basis>:<beta>"``.
    :return: Dense ``(n_basis, n_basis)`` matrix in PySCF AO order, float64.
    """
    return compute_metric_matrix(system, aux_basis, "coulomb")


# ── Packing ───────────────────────────────────────────────────────────────────


def pack_metric_matrices(matrices: List[torch.Tensor]) -> TensorMap:
    """
    Pack per-system metric matrices into a TensorMap, one block per system.

    Systems in a batch differ in size -- anything from a single atom to hundreds --
    so the matrices are stored **ragged**, as one block each, rather than padded into
    a single ``(n_systems, n_max, n_max)`` array. Padding costs
    ``n_systems * n_max**2`` against the ``sum_i n_i**2`` actually needed, which on a
    batch of mostly small systems with a few large ones measured ~8x more memory and
    transport, and can exhaust host memory outright. A TensorMap is already a
    collection of differently shaped blocks, so this needs no special transport.

    :param matrices: One dense square matrix per system.
    :return: TensorMap keyed by ``system``, block ``i`` holding ``(n_i, n_i)``.
    """
    if len(matrices) == 0:
        raise ValueError("expected at least one metric matrix to pack")

    device = matrices[0].device
    blocks = []
    for i_system, matrix in enumerate(matrices):
        basis = torch.arange(matrix.shape[0], dtype=torch.int32, device=device)
        blocks.append(
            TensorBlock(
                values=matrix,
                # The samples carry a "system" dimension because the O(3) augmenter
                # requires one to route rows to transformations. There are no
                # component axes, so it recognises the block as invariant and passes
                # it through untouched -- which is what allows the metric to be built
                # on the unaugmented geometry, before augmentation runs.
                samples=Labels(
                    names=["system", "basis"],
                    values=torch.stack(
                        [torch.full_like(basis, i_system), basis], dim=1
                    ),
                ),
                components=[],
                properties=Labels(names=["basis_2"], values=basis.reshape(-1, 1)),
            )
        )
    keys = Labels(
        names=["system"],
        values=torch.arange(len(matrices), dtype=torch.int32, device=device).reshape(
            -1, 1
        ),
    )
    return TensorMap(keys, blocks)


def unpack_metric_matrices(packed: TensorMap) -> List[torch.Tensor]:
    """
    Recover the per-system metric matrices.

    :param packed: Output of :func:`pack_metric_matrices`.
    :return: One ``(n_i, n_i)`` matrix per system, in batch order.
    """
    return [packed.block(i).values for i in range(len(packed))]


# ── Collate transforms ────────────────────────────────────────────────────────


def _metric_matrices_transform(
    target_to_aux_basis: Mapping[str, str],
    metric: str,
    systems: List[System],
    targets: Dict[str, TensorMap],
    extra: Dict[str, TensorMap],
) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
    packed_by_basis: Dict[str, TensorMap] = {}
    for target_name, aux_basis in target_to_aux_basis.items():
        if aux_basis not in packed_by_basis:
            packed_by_basis[aux_basis] = pack_metric_matrices(
                [compute_metric_matrix(system, aux_basis, metric) for system in systems]
            )
        extra[metric_matrix_name(target_name, metric)] = packed_by_basis[aux_basis]
    return systems, targets, extra


def get_metric_matrices_transform(
    target_to_aux_basis: Mapping[str, str],
    metric: str,
) -> Callable:
    """
    Build a collate transform attaching per-target two-centre metric matrices.

    **This transform must run before the augmenter.** The metric depends on the
    geometry, so it is computed in the dataset's own orientation, and the losses are
    evaluated in that frame (see
    :attr:`~metatrain.utils.loss.LossInterface.evaluate_in_original_frame`) rather
    than the augmented one.

    The matrices are recomputed for every batch. They depend only on the unaugmented
    geometry, so they could in principle be cached across epochs, but doing that well
    is a memory- and dataloader-topology problem rather than a scientific one, and is
    deliberately left out of this module.

    Targets sharing an auxiliary basis share one computation per batch.

    :param target_to_aux_basis: Mapping from target name to auxiliary basis name.
    :param metric: ``"overlap"`` or ``"coulomb"``.
    :return: A collate transform.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}.")
    return functools.partial(
        _metric_matrices_transform, dict(target_to_aux_basis), metric
    )


def resolve_aux_basis(
    target_name: str, aux_basis: Union[str, Mapping[str, str]]
) -> str:
    """
    Resolve the auxiliary basis configured for one target.

    :param target_name: Name of the RI-coefficient target.
    :param aux_basis: Either a basis name applying to every target, or a mapping
        from target name to basis name.
    :return: The basis name for this target.
    """
    if isinstance(aux_basis, str):
        return aux_basis
    if target_name in aux_basis:
        return aux_basis[target_name]
    raise ValueError(
        f"no auxiliary basis configured for target '{target_name}'; "
        f"available targets: {', '.join(sorted(aux_basis))}."
    )
