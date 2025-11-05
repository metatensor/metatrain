import sys

import metatensor.torch as mts
import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from scipy.spatial.transform import Rotation

from metatrain.utils.augmentation import RotationalAugmenter
from metatrain.utils.data import DatasetInfo, DiskDataset, TargetInfo
from metatrain.utils.data.target_info import get_generic_target_info

from . import RESOURCES_PATH


@pytest.fixture
def layout_spherical():
    return TensorMap(
        keys=Labels(
            names=["o3_lambda", "o3_sigma"],
            values=torch.tensor([[0, 1], [2, 1]]),
        ),
        blocks=[
            TensorBlock(
                values=torch.empty(0, 1, 1),
                samples=Labels(
                    names=["system"],
                    values=torch.empty((0, 1), dtype=torch.int32),
                ),
                components=[
                    Labels(
                        names=["o3_mu"],
                        values=torch.arange(0, 1, dtype=torch.int32).reshape(-1, 1),
                    ),
                ],
                properties=Labels.single(),
            ),
            TensorBlock(
                values=torch.empty(0, 5, 1),
                samples=Labels(
                    names=["system"],
                    values=torch.empty((0, 1), dtype=torch.int32),
                ),
                components=[
                    Labels(
                        names=["o3_mu"],
                        values=torch.arange(-2, 3, dtype=torch.int32).reshape(-1, 1),
                    ),
                ],
                properties=Labels.single(),
            ),
        ],
    )


@pytest.mark.parametrize("batch_size", [1, 2])
def test_rotation_per_structure_spherical(batch_size):
    """Tests that the rotational augmenter rotates a dipole moment consistent with
    targets computed from DFT"""

    target_name = "mtt::dipole_moment"

    # Hard-coded rotation matrix
    R = np.array(
        [
            [0.22922512, -0.15149287, 0.96151222],
            [0.66175278, -0.70015582, -0.26807664],
            [0.71382008, 0.69773328, -0.06024247],
        ]
    )

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
        "samples",
        remove_tensor_name=True,
    )
    fRX = mts.join(
        [
            sample[target_name]
            for i, sample in enumerate(dataset_rotated)
            if i < batch_size
        ],
        "samples",
        remove_tensor_name=True,
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
                    "per_atom": False,
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
        extra_data={},
        rotations=[Rotation.from_matrix(R.T)] * batch_size,
        inversions=[1] * batch_size,
    )
    RfX = RfX[target_name]

    # Check that the rotated target matches the reference
    mts.allclose_raise(RfX, fRX, atol=1e-5)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_rotation_per_atom_spherical(batch_size):
    """Tests that the rotational augmenter rotates a dipole moment consistent with
    targets computed from DFT"""

    target_name = "mtt::electron_density_basis_projs"

    # Hard-coded rotation matrix used to generate the system for which DFT was run
    R = np.array(
        [
            [0.22922512, -0.15149287, 0.96151222],
            [0.66175278, -0.70015582, -0.26807664],
            [0.71382008, 0.69773328, -0.06024247],
        ]
    )

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
        "samples",
        remove_tensor_name=True,
    )
    fRX = mts.join(
        [
            sample[target_name]
            for i, sample in enumerate(dataset_rotated)
            if i < batch_size
        ],
        "samples",
        remove_tensor_name=True,
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
                    "per_atom": True,
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
        extra_data={},
        rotations=[Rotation.from_matrix(R.T)] * batch_size,
        inversions=[1] * batch_size,
    )
    RfX = RfX[target_name]

    # Check that the rotated target matches the reference
    mts.allclose_raise(RfX, fRX, atol=1e-5)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_rotation_per_pair_spherical(batch_size):
    """Tests that the rotational augmenter rotates a Hamiltonian matrix consistent with
    targets computed from DFT"""

    target_name = "mtt::hamiltonian"

    # Hard-coded rotation matrix used to generate the system for which DFT was run
    R = np.array(
        [
            [0.22922512, -0.15149287, 0.96151222],
            [0.66175278, -0.70015582, -0.26807664],
            [0.71382008, 0.69773328, -0.06024247],
        ]
    )

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
        "samples",
        remove_tensor_name=True,
    )
    fRX = mts.join(
        [
            sample[target_name]
            for i, sample in enumerate(dataset_rotated)
            if i < batch_size
        ],
        "samples",
        remove_tensor_name=True,
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
                                {"o3_lambda": 0, "o3_sigma": 1, "n_centers": 1},
                                {"o3_lambda": 0, "o3_sigma": 1, "n_centers": 2},
                                {"o3_lambda": 1, "o3_sigma": 1, "n_centers": 2},
                                {"o3_lambda": 1, "o3_sigma": 1, "n_centers": 1},
                                {"o3_lambda": 1, "o3_sigma": -1, "n_centers": 2},
                                {"o3_lambda": 2, "o3_sigma": 1, "n_centers": 1},
                                {"o3_lambda": 2, "o3_sigma": 1, "n_centers": 2},
                            ]
                        }
                    },
                    "per_atom": True,
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
        extra_data={},
        rotations=[Rotation.from_matrix(R.T)] * batch_size,
        inversions=[1] * batch_size,
    )
    RfX = RfX[target_name]

    # Check that the rotated target matches the reference

    for k in RfX.keys:
        block_RfX = RfX.block(k)
        block_fRX = fRX.block(k)

        assert torch.equal(torch.isnan(block_RfX.values), torch.isnan(block_fRX.values))
        mask = ~torch.isnan(block_RfX.values)
        assert torch.allclose(block_RfX.values[mask], block_fRX.values[mask])


def test_missing_library(monkeypatch, layout_spherical):
    # Pretend 'spherical' is not installed
    monkeypatch.setitem(sys.modules, "spherical", None)

    target_info_dict = {
        "foo": TargetInfo(quantity="energy", unit=None, layout=layout_spherical)
    }

    msg = (
        "To perform data augmentation on spherical targets, please "
        "install the `spherical` package with `pip install spherical`."
    )
    with pytest.raises(ImportError, match=msg):
        RotationalAugmenter(target_info_dict)
