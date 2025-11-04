import random

import ase
import metatensor.torch as mts
import numpy as np
import pytest
import torch
from metatomic.torch import ModelOutput, systems_to_torch

from metatrain.pet import PET
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import TargetInfo
from metatrain.utils.matrix import Blocks2Matrix
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import (
    LAYOUT_HAM_PATH,
    MODEL_HYPERS,
)


def get_frames():
    """
    Returns 3 physically equivalent CH4 systems with different atom orderings.

    The second is the same as the first but with atoms 3 (H) and 4 (H) swapped.
    The third is the same as the first but with atoms 1 (C) and 2 (H) swapped.
    """
    return [
        ase.Atoms(
            numbers=[6, 1, 1, 1, 1],
            positions=[
                [0.99826, -0.00246, -0.00436],
                [2.09021, -0.00243, 0.00414],
                [0.63379, 1.02686, 0.00414],
                [0.62704, -0.52773, 0.87811],
                [0.64136, -0.50747, -0.9054],
            ],
        ),
        ase.Atoms(
            numbers=[6, 1, 1, 1, 1],
            positions=[
                [0.99826, -0.00246, -0.00436],
                [2.09021, -0.00243, 0.00414],
                [0.63379, 1.02686, 0.00414],
                [0.64136, -0.50747, -0.9054],  # swapped
                [0.62704, -0.52773, 0.87811],  # swapped
            ],
        ),
        ase.Atoms(
            numbers=[1, 6, 1, 1, 1],
            positions=[
                [2.09021, -0.00243, 0.00414],  # swapped
                [0.99826, -0.00246, -0.00436],  # swapped
                [0.63379, 1.02686, 0.00414],
                [0.62704, -0.52773, 0.87811],
                [0.64136, -0.50747, -0.9054],
            ],
        ),
    ]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matrix_hermitian(dtype):
    """Test that the matrix predicted is Hermitian"""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    targets = {}
    targets["mtt::hamiltonian"] = TargetInfo(
        quantity="",
        layout=mts.load(LAYOUT_HAM_PATH),
        unit="",
    )
    basis_set = {(0, 1): 2, (0, 6): 3, (1, 6): 2}

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6], targets=targets
    )
    model = PET(MODEL_HYPERS, dataset_info).to(dtype)

    # Predict on the first five systems
    frames = get_frames()
    systems = systems_to_torch(frames)
    systems = [system.to(dtype) for system in systems]
    for system in systems:
        get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

    output = model(
        systems,
        {"mtt::hamiltonian": ModelOutput(quantity="", unit="", per_atom=True)},
    )["mtt::hamiltonian"]

    b2m = Blocks2Matrix(
        basis_set=basis_set,  # 6-31G
        o3_lambda_max=1,
        dtype=systems[0].positions.dtype,
        device=systems[0].positions.device,
    )
    output_matrices = b2m(
        systems,
        tensor=output,
    )
    for mat in output_matrices:
        assert torch.allclose(mat[(0, 0, 0)], mat[(0, 0, 0)].conj().T)
        print((mat[(0, 0, 0)] - mat[(0, 0, 0)].conj().T).abs().mean())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matrix_atom_swapping(dtype):
    """Test that the matrix predicted is Hermitian"""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    targets = {}
    targets["mtt::hamiltonian"] = TargetInfo(
        quantity="",
        layout=mts.load(LAYOUT_HAM_PATH),
        unit="",
    )
    basis_set = {(0, 1): 2, (0, 6): 3, (1, 6): 2}

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6], targets=targets
    )
    model = PET(MODEL_HYPERS, dataset_info).to(dtype)

    # Predict on the first five systems
    frames = get_frames()
    systems = systems_to_torch(frames)
    systems = [system.to(dtype) for system in systems]
    for system in systems:
        get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

    output = model(
        systems,
        {"mtt::hamiltonian": ModelOutput(quantity="", unit="", per_atom=True)},
    )["mtt::hamiltonian"]

    b2m = Blocks2Matrix(
        basis_set=basis_set,  # 6-31G
        o3_lambda_max=1,
        dtype=systems[0].positions.dtype,
        device=systems[0].positions.device,
    )
    output_matrices = b2m(
        systems,
        tensor=output,
    )

    # Check H H permutation
    mat1 = output_matrices[0][0, 0, 0].detach().numpy()
    mat2 = _permute_atomic_subblocks(
        output_matrices[1][0, 0, 0].detach().numpy(),
        frames[1],
        basis_set,
        [0, 1, 2, 4, 3],
    )
    mat3 = _permute_atomic_subblocks(
        output_matrices[2][0, 0, 0].detach().numpy(),
        frames[2],
        basis_set,
        [1, 0, 2, 3, 4],
    )
    print("H H swap diff mean:", np.abs(mat1 - mat2).mean())
    print("C H swap diff mean:", np.abs(mat1 - mat3).mean())

    if dtype == torch.float32:
        rtol = 1e-5
        atol = 1e-7
    else:
        rtol = 1e-10
        atol = 1e-12

    assert np.allclose(mat1, mat2, rtol=rtol, atol=atol)
    assert np.allclose(mat1, mat3, rtol=rtol, atol=atol)


def _permute_atomic_subblocks(matrix, frame, basis_set, permutation):
    # split matrix by atoms blocks
    split_idx = np.cumsum(
        [
            sum(
                (2 * ell + 1) * radial
                for (ell, Z), radial in basis_set.items()
                if Z == Z_atom
            )
            for Z_atom in frame.numbers
        ]
    )[:-1]

    split_mat = [
        np.split(mat, split_idx, axis=1) for mat in np.split(matrix, split_idx, axis=0)
    ]

    # permute blocks
    return np.block([[split_mat[i][j] for j in permutation] for i in permutation])
