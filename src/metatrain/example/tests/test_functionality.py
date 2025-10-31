"""Basic functionality tests for the example ZeroModel architecture."""

import torch
from metatomic.torch import System

from metatrain.example import ZeroModel, ZeroTrainer
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)


DEFAULT_HYPERS = {
    "model": {
        "cutoff": 5.0,
    },
    "training": {
        "distributed": False,
        "distributed_port": 39591,
        "batch_size": 2,
        "num_epochs": 2,
        "warmup_fraction": 0.01,
        "learning_rate": 0.001,
        "log_interval": 1,
        "checkpoint_interval": 1,
        "scale_targets": True,
        "fixed_composition_weights": {},
        "fixed_scaling_weights": {},
        "per_structure_targets": [],
        "num_workers": 0,
        "log_mae": False,
        "log_separate_blocks": False,
        "best_model_metric": "rmse_prod",
        "loss": "mse",
    },
}


def test_zero_energy_prediction():
    """Test that the ZeroModel always predicts zero energy."""
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
    )

    model = ZeroModel(DEFAULT_HYPERS, dataset_info)

    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )

    requested_neighbor_lists = get_requested_neighbor_lists(model)
    system = get_system_with_neighbor_lists(system, requested_neighbor_lists)

    # Get predictions
    predictions = model(
        [system],
        {"energy": model.outputs["energy"]},
    )

    # Check that energy is zero (within numerical precision)
    energy_block = predictions["energy"].block()
    assert torch.allclose(
        energy_block.values, torch.zeros_like(energy_block.values), atol=1e-6
    )


def test_trainer_no_rotational_augmentation():
    """Test that the ZeroTrainer does not use rotational augmentation."""
    trainer = ZeroTrainer(DEFAULT_HYPERS["training"])
    assert trainer.use_rotational_augmentation() is False


def test_multiple_systems():
    """Test that the ZeroModel can handle multiple systems."""
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
    )

    model = ZeroModel(DEFAULT_HYPERS, dataset_info)

    system1 = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )

    system2 = System(
        types=torch.tensor([1, 1, 1]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )

    requested_neighbor_lists = get_requested_neighbor_lists(model)
    system1 = get_system_with_neighbor_lists(system1, requested_neighbor_lists)
    system2 = get_system_with_neighbor_lists(system2, requested_neighbor_lists)

    # Get predictions for multiple systems
    predictions = model(
        [system1, system2],
        {"energy": model.outputs["energy"]},
    )

    # Check that energies are zero for both systems
    energy_block = predictions["energy"].block()
    assert energy_block.values.shape[0] == 2  # Two systems
    assert torch.allclose(
        energy_block.values, torch.zeros_like(energy_block.values), atol=1e-6
    )
