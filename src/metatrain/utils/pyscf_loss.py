from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

if TYPE_CHECKING:
    from types import ModuleType

    from pyscf.gto import Mole


RIAuxBasis = str | dict[str, str]


def _import_pyscf_modules() -> tuple[ModuleType, ModuleType]:
    try:
        gto = importlib.import_module("pyscf.gto")
        elements = importlib.import_module("pyscf.data.elements")
    except ModuleNotFoundError as err:
        raise ImportError(
            "RI Coulomb losses require `pyscf` to compute auxiliary Coulomb matrices."
        ) from err

    return gto, elements


def coulomb_matrix_name(target_name: str) -> str:
    """Return the ``extra_data`` key used for a target's Coulomb matrix."""

    return f"{target_name}_coulomb_matrix"


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

    The molecule is only used for integral evaluation, so we let PySCF infer a
    spin-compatible electron count automatically.

    :param system: Atomistic system with positions in Angstrom.
    :param aux_basis: PySCF auxiliary basis name.
    :return: Built PySCF molecule in spherical-harmonic form.
    """

    gto, _ = _import_pyscf_modules()

    mol = gto.Mole()
    mol.atom = _system_to_atom_string(system)
    mol.basis = aux_basis
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.spin = None
    mol.cart = False
    mol.build()
    return mol


def compute_coulomb_matrix(system: System, aux_basis: str) -> torch.Tensor:
    """
    Compute the two-center Coulomb matrix for one system and auxiliary basis.

    :param system: Atomistic system with positions in Angstrom.
    :param aux_basis: PySCF auxiliary basis name.
    :return: Dense Coulomb matrix ``J`` in PySCF AO order.
    """

    auxmol = build_auxiliary_molecule(system, aux_basis)
    return torch.from_numpy(auxmol.intor("int2c2e")).to(torch.float64)


def pack_coulomb_matrices(matrices: list[torch.Tensor]) -> TensorMap:
    """
    Pack variable-size per-system Coulomb matrices into one padded ``TensorMap``.

    :param matrices: Dense square matrices, one per system.
    :return: Single-block ``TensorMap`` with samples ``system``.
    """

    if len(matrices) == 0:
        raise ValueError("Expected at least one Coulomb matrix to pack.")

    max_basis = max(matrix.shape[0] for matrix in matrices)
    dtype = matrices[0].dtype
    device = matrices[0].device

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
            names=["system"],
            values=torch.arange(
                len(matrices), dtype=torch.int32, device=device
            ).reshape(-1, 1),
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


def unpack_coulomb_matrices(
    batched_matrices: TensorMap, basis_sizes: list[int]
) -> list[torch.Tensor]:
    """
    Unpack a padded batch of Coulomb matrices back to dense per-system matrices.

    :param batched_matrices: Padded batched Coulomb matrices.
    :param basis_sizes: Physical basis size for each system.
    :return: Dense Coulomb matrices cropped to each system size.
    """

    block = batched_matrices.block()
    if len(block.samples) != len(basis_sizes):
        raise ValueError(
            "The number of packed Coulomb matrices does not match the basis sizes."
        )

    for i_system, n_basis in enumerate(basis_sizes):
        matrix = block.values[i_system]
        valid_rows = ~torch.isnan(matrix).all(dim=1)
        actual_basis_size = int(valid_rows.sum().item())

        if n_basis > actual_basis_size:
            raise ValueError(
                "At least one requested Coulomb matrix basis size exceeds the packed "
                "matrix size."
            )
        if n_basis != actual_basis_size:
            raise ValueError(
                "The RI target size does not match the configured auxiliary basis. "
                "Check that 'ri_aux_basis' matches the RI coefficients stored in the "
                "dataset."
            )

    return [
        block.values[i_system, :n_basis, :n_basis]
        for i_system, n_basis in enumerate(basis_sizes)
    ]


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
    return pack_coulomb_matrices(matrices)


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
