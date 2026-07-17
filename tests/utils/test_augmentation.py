import metatensor.torch as mts
import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatrain.utils.augmentation import RotationalAugmenter
from metatrain.utils.data import DatasetInfo, DiskDataset, TargetInfo
from metatrain.utils.data.target_info import get_generic_target_info

from ..conftest import RESOURCES_PATH


# Hard-coded rotation matrix used to generate the rotated DFT dataset
_R = np.array(
    [
        [0.22922512, -0.15149287, 0.96151222],
        [0.66175278, -0.70015582, -0.26807664],
        [0.71382008, 0.69773328, -0.06024247],
    ]
)


def _transformation(batch_size: int) -> list:
    # apply_augmentations expects List[torch.Tensor] where each matrix R satisfies
    # positions_rotated = positions @ R.T, matching how the test datasets were built
    t = torch.tensor(_R.T, dtype=torch.float64)
    return [t] * batch_size


@pytest.mark.parametrize("batch_size", [1, 2])
def test_rotation_per_structure_spherical(batch_size):
    """Tests that the rotational augmenter rotates a dipole moment consistent with
    targets computed from DFT"""

    target_name = "mtt::dipole_moment"

    # Load the target data
    dataset_unrotated = DiskDataset(RESOURCES_PATH / "spherical_targets_unrotated.zip")
    dataset_rotated = DiskDataset(RESOURCES_PATH / "spherical_targets_rotated.zip")
    X = [
        sample["system"].to(torch.float64)
        for i, sample in enumerate(dataset_unrotated)
        if i < batch_size
    ]
    fX = mts.join(
        [
            sample[target_name]
            for i, sample in enumerate(dataset_unrotated)
            if i < batch_size
        ],
        axis="samples",
    )
    fRX = mts.join(
        [
            sample[target_name]
            for i, sample in enumerate(dataset_rotated)
            if i < batch_size
        ],
        axis="samples",
    )

    # Init the RotationalAugmenter
    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 8],
        targets={
            target_name: get_generic_target_info(
                target_name,
                {
                    "quantity": "spherical",
                    "unit": "",
                    "type": {
                        "spherical": {
                            "irreps": [
                                {"o3_lambda": 1, "o3_sigma": 1},  # dipole
                            ]
                        }
                    },
                    "sample_kind": "system",
                    "num_subtargets": 1,
                },
            )
        },
    )
    rotational_augmenter = RotationalAugmenter(dataset_info.targets, {})

    # Apply the augmentation to the target
    _, RfX, _ = rotational_augmenter.apply_augmentations(
        X,
        {target_name: fX},
        _transformation(batch_size),
        extra_data={},
    )
    RfX = RfX[target_name]

    # Check that the rotated target matches the reference
    mts.allclose_raise(RfX, fRX, atol=1e-5)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_rotation_per_atom_spherical(batch_size):
    """Tests that the rotational augmenter rotates electron density projections
    consistent with targets computed from DFT"""

    target_name = "mtt::electron_density_basis_projs"

    # Load the target data
    dataset_unrotated = DiskDataset(RESOURCES_PATH / "spherical_targets_unrotated.zip")
    dataset_rotated = DiskDataset(RESOURCES_PATH / "spherical_targets_rotated.zip")
    X = [
        sample["system"].to(torch.float64)
        for i, sample in enumerate(dataset_unrotated)
        if i < batch_size
    ]
    fX = mts.join(
        [
            sample[target_name]
            for i, sample in enumerate(dataset_unrotated)
            if i < batch_size
        ],
        axis="samples",
    )
    fRX = mts.join(
        [
            sample[target_name]
            for i, sample in enumerate(dataset_rotated)
            if i < batch_size
        ],
        axis="samples",
    )

    # Init the RotationalAugmenter
    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 8],
        targets={
            target_name: get_generic_target_info(
                target_name,
                {
                    "quantity": "spherical",
                    "unit": "",
                    "type": {
                        "spherical": {
                            "irreps": [
                                {"o3_lambda": 0, "o3_sigma": 1},
                                {"o3_lambda": 1, "o3_sigma": 1},
                                {"o3_lambda": 2, "o3_sigma": 1},
                                {"o3_lambda": 3, "o3_sigma": 1},
                            ]
                        }
                    },
                    "sample_kind": "atom",
                    "num_subtargets": 1,
                },
            )
        },
    )
    rotational_augmenter = RotationalAugmenter(dataset_info.targets, {})

    # Apply the augmentation to the target
    _, RfX, _ = rotational_augmenter.apply_augmentations(
        X,
        {target_name: fX},
        _transformation(batch_size),
        extra_data={},
    )
    RfX = RfX[target_name]

    # Check that the rotated target matches the reference
    mts.allclose_raise(RfX, fRX, atol=1e-5)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_rotation_per_atom_spherical_atomicbasis(batch_size):
    """Tests that the rotational augmenter rotates a Hamiltonian in the coupled basis
    (rank-1 tensors with an atomic basis) consistent with targets computed from DFT.

    Previously this raised ValueError; metatomic's apply_augmentations now handles
    atomic basis targets via per-block row-index indexing.
    """
    target_name = "mtt::hamiltonian_nodes"

    # Load the target data
    dataset_unrotated = DiskDataset(RESOURCES_PATH / "spherical_targets_unrotated.zip")
    dataset_rotated = DiskDataset(RESOURCES_PATH / "spherical_targets_rotated.zip")

    X = [
        sample["system"].to(torch.float64)
        for i, sample in enumerate(dataset_unrotated)
        if i < batch_size
    ]
    fX = mts.join(
        [
            sample[target_name]
            for i, sample in enumerate(dataset_unrotated)
            if i < batch_size
        ],
        axis="samples",
    )
    fRX = mts.join(
        [
            sample[target_name]
            for i, sample in enumerate(dataset_rotated)
            if i < batch_size
        ],
        axis="samples",
    )

    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 8],
        targets={
            target_name: get_generic_target_info(
                target_name,
                {
                    "quantity": "spherical",
                    "unit": "",
                    "type": {
                        "spherical": {
                            "product": "coupled",
                            "irreps": {
                                1: [  # H
                                    {"o3_lambda": 0, "o3_sigma": 1, "num": 3},
                                    {"o3_lambda": 1, "o3_sigma": 1, "num": 1},
                                ],
                                8: [  # O
                                    {"o3_lambda": 0, "o3_sigma": 1, "num": 5},
                                    {"o3_lambda": 1, "o3_sigma": 1, "num": 3},
                                    {"o3_lambda": 2, "o3_sigma": 1, "num": 2},
                                    {"o3_lambda": 3, "o3_sigma": 1, "num": 1},
                                ],
                            },
                        }
                    },
                    "sample_kind": "atom",
                    "num_subtargets": 1,
                },
            )
        },
    )

    rotational_augmenter = RotationalAugmenter(dataset_info.targets, {})

    # Apply the augmentation to the target
    _, RfX, _ = rotational_augmenter.apply_augmentations(
        X,
        {target_name: fX},
        _transformation(batch_size),
        extra_data={},
    )
    RfX = RfX[target_name]

    # Check that the rotated target matches the reference
    mts.allclose_raise(RfX, fRX, atol=1e-5)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_rotation_per_atom_spherical_rank2(batch_size):
    """Tests that the rotational augmenter rotates a Hamiltonian in the uncoupled basis
    (rank-2 tensors with an atomic basis) consistent with targets computed from DFT.

    Previously this raised ValueError; metatomic's apply_augmentations now handles
    atomic basis targets via per-block row-index indexing.
    """
    target_name = "mtt::hamiltonian_nodes_uncoupled"

    # Load the target data
    dataset_unrotated = DiskDataset(RESOURCES_PATH / "spherical_targets_unrotated.zip")
    dataset_rotated = DiskDataset(RESOURCES_PATH / "spherical_targets_rotated.zip")

    X = [
        sample["system"].to(torch.float64)
        for i, sample in enumerate(dataset_unrotated)
        if i < batch_size
    ]
    fX = mts.join(
        [
            sample[target_name]
            for i, sample in enumerate(dataset_unrotated)
            if i < batch_size
        ],
        axis="samples",
    )
    fRX = mts.join(
        [
            sample[target_name]
            for i, sample in enumerate(dataset_rotated)
            if i < batch_size
        ],
        axis="samples",
    )

    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 8],
        targets={
            target_name: get_generic_target_info(
                target_name,
                {
                    "quantity": "spherical",
                    "unit": "",
                    "type": {
                        "spherical": {
                            "product": "cartesian",
                            "irreps": {
                                1: [  # H
                                    {"o3_lambda": 0, "o3_sigma": 1, "num": 3},
                                    {"o3_lambda": 1, "o3_sigma": 1, "num": 1},
                                ],
                                8: [  # O
                                    {"o3_lambda": 0, "o3_sigma": 1, "num": 5},
                                    {"o3_lambda": 1, "o3_sigma": 1, "num": 3},
                                    {"o3_lambda": 2, "o3_sigma": 1, "num": 2},
                                    {"o3_lambda": 3, "o3_sigma": 1, "num": 1},
                                ],
                            },
                        }
                    },
                    "sample_kind": "atom",
                    "num_subtargets": 1,
                },
            )
        },
    )

    rotational_augmenter = RotationalAugmenter(dataset_info.targets, {})

    # Apply the augmentation to the target
    _, RfX, _ = rotational_augmenter.apply_augmentations(
        X,
        {target_name: fX},
        _transformation(batch_size),
        extra_data={},
    )
    RfX = RfX[target_name]

    # Check that the rotated target matches the reference
    mts.allclose_raise(RfX, fRX, atol=1e-5)
