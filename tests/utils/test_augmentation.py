import sys

import metatensor.torch as mts
import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from metatrain.utils.augmentation import RotationalAugmenter
from metatrain.utils.data import DatasetInfo, DiskDataset, TargetInfo
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)

from ..conftest import RESOURCES_PATH


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
        extra_data={},
        rotations=[Rotation.from_matrix(R.T)] * batch_size,
        inversions=[1] * batch_size,
    )
    RfX = RfX[target_name]

    # Check that the rotated target matches the reference
    mts.allclose_raise(RfX, fRX, atol=1e-5)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_rotation_per_atom_spherical(batch_size):
    """Tests that the rotational augmenter rotates electron density projections
    consistent with targets computed from DFT"""

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
        extra_data={},
        rotations=[Rotation.from_matrix(R.T)] * batch_size,
        inversions=[1] * batch_size,
    )
    RfX = RfX[target_name]

    # Check that the rotated target matches the reference
    mts.allclose_raise(RfX, fRX, atol=1e-5)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_rotation_per_atom_spherical_atomicbasis(batch_size):
    """Tests that the rotational augmenter rotates a Hamiltonian
    in the coupled basis (rank 1 tensors with an atomic basis)
    consistent with targets computed from DFT"""
    target_name = "mtt::hamiltonian_nodes"

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

    with pytest.raises(
        (ValueError, torch.jit.Error),
        match="Rotational augmentation of atomic basis targets is not supported yet.",
    ):
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
def test_rotation_per_atom_spherical_rank2(batch_size):
    """Tests that the rotational augmenter rotates a Hamiltonian
    in the uncoupled basis (rank 2 tensors with an atomic basis)
    consistent with targets computed from DFT"""
    target_name = "mtt::hamiltonian_nodes_uncoupled"

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

    with pytest.raises(
        (ValueError, torch.jit.Error),
        match="Rotational augmentation of atomic basis targets is not supported yet.",
    ):
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


def test_augmentation_does_not_rotate_loss_weights():
    """Per-sample loss weights (``*_weights`` extra_data) must be left untouched by the
    rotational augmenter, even when they carry Cartesian (xyz) gradient components."""
    n_atoms = 4

    energy_info = get_energy_target_info(
        "energy",
        OmegaConf.create({"quantity": "energy", "unit": "eV"}),
        add_position_gradients=True,
    )
    # The weights mirror the structure of the energy target.
    weights_info = TargetInfo(layout=energy_info.layout, quantity="", unit="")

    augmenter = RotationalAugmenter(
        {"energy": energy_info}, {"energy_weights": weights_info}
    )

    system = System(
        types=torch.ones(n_atoms, dtype=torch.int32),
        positions=torch.rand(n_atoms, 3, dtype=torch.float64),
        cell=torch.zeros((3, 3), dtype=torch.float64),
        pbc=torch.zeros(3, dtype=torch.bool),
    )

    def make_energy_like_tmap(energy_value, gradient_values):
        block = TensorBlock(
            values=torch.tensor([[energy_value]], dtype=torch.float64),
            samples=Labels(["system"], torch.tensor([[0]])),
            components=[],
            properties=Labels.range("energy", 1),
        )
        block.add_gradient(
            "positions",
            TensorBlock(
                values=gradient_values,
                samples=Labels(
                    ["sample", "atom"],
                    torch.tensor([[0, a] for a in range(n_atoms)]),
                ),
                components=[Labels(["xyz"], torch.arange(3).reshape(-1, 1))],
                properties=Labels.range("energy", 1),
            ),
        )
        return TensorMap(keys=Labels.single(), blocks=[block])

    forces = torch.arange(1, n_atoms * 3 + 1, dtype=torch.float64).reshape(
        n_atoms, 3, 1
    )
    energy_tm = make_energy_like_tmap(1.5, forces.clone())

    # distinct xyz weights so that, if they were (wrongly) rotated, they would change
    weight_grad = torch.arange(1, n_atoms * 3 + 1, dtype=torch.float64).reshape(
        n_atoms, 3, 1
    )
    weights_tm = make_energy_like_tmap(2.0, weight_grad.clone())

    rotation = Rotation.from_euler("xyz", [30.0, 45.0, 60.0], degrees=True)
    _, new_targets, new_extra_data = augmenter.apply_augmentations(
        [system],
        {"energy": energy_tm},
        rotations=[rotation],
        inversions=[1],
        extra_data={"energy_weights": weights_tm},
    )

    # Sanity check: the target forces were actually rotated (so the test is meaningful)
    assert not torch.allclose(
        new_targets["energy"].block().gradient("positions").values, forces
    )

    # The weights must be unchanged, both for the values and the gradient
    new_weights = new_extra_data["energy_weights"].block()
    torch.testing.assert_close(
        new_weights.values, torch.tensor([[2.0]], dtype=torch.float64)
    )
    torch.testing.assert_close(new_weights.gradient("positions").values, weight_grad)


def test_missing_library(monkeypatch, layout_spherical):
    # Pretend 'spherical' is not installed
    monkeypatch.setitem(sys.modules, "spherical", None)

    target_info_dict = {
        "foo": TargetInfo(layout=layout_spherical, quantity="energy", unit="")
    }

    msg = (
        "To perform data augmentation on spherical targets, please "
        "install the `spherical` package with `pip install spherical`."
    )
    with pytest.raises(ImportError, match=msg):
        RotationalAugmenter(target_info_dict)
