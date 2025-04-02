import pytest


pytest.importorskip("torchpme")

import copy

import torch
from metatensor.torch.atomistic import ModelOutput, System
from omegaconf import OmegaConf

from metatrain.experimental.nativepet import NativePET, Trainer
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import (
    read_systems,
    read_targets,
)
from metatrain.utils.data.target_info import (
    get_energy_target_info,
)
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DATASET_WITH_FORCES_PATH, DEFAULT_HYPERS, MODEL_HYPERS


@pytest.mark.parametrize("use_ewald", [True, False])
def test_long_range_features(use_ewald):
    """Tests that long-range features can be computed."""
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info({"quantity": "energy", "unit": "eV"})
        },
    )
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["long_range"]["enable"] = True
    hypers["long_range"]["use_ewald"] = use_ewald
    model = NativePET(hypers, dataset_info)

    system = System(
        types=torch.tensor([6, 6, 8, 8]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]
        ),
        cell=torch.eye(3) * 10,
        pbc=torch.tensor([True, True, True]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    outputs = {"energy": ModelOutput(per_atom=False)}
    model([system, system], outputs)


@pytest.mark.parametrize("use_ewald", [True, False])
def test_long_range_training(use_ewald):
    """Tests that long-range features can be computed."""
    pytest.importorskip("torch", minversion="1.20")
    systems = read_systems(DATASET_WITH_FORCES_PATH)

    conf = {
        "energy": {
            "quantity": "energy",
            "read_from": DATASET_WITH_FORCES_PATH,
            "reader": "ase",
            "key": "energy",
            "unit": "eV",
            "type": "scalar",
            "per_atom": False,
            "num_subtargets": 1,
            "forces": {"read_from": DATASET_WITH_FORCES_PATH, "key": "force"},
            "stress": False,
            "virial": False,
        }
    }

    targets, target_info_dict = read_targets(OmegaConf.create(conf))
    targets = {"energy": targets["energy"]}
    dataset = Dataset.from_dict({"system": systems, "energy": targets["energy"]})
    hypers = DEFAULT_HYPERS.copy()
    hypers["training"]["num_epochs"] = 2
    hypers["training"]["scheduler_patience"] = 1
    hypers["training"]["fixed_composition_weights"] = {}

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[6], targets=target_info_dict
    )

    model_hypers = copy.deepcopy(MODEL_HYPERS)
    model_hypers["long_range"]["enable"] = True
    model_hypers["long_range"]["use_ewald"] = use_ewald
    model = NativePET(model_hypers, dataset_info)

    trainer = Trainer(hypers["training"])
    trainer.train(
        model=model,
        dtype=torch.float32,
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir=".",
    )

    # Predict on the first five systems
    systems = [system.to(torch.float32) for system in systems]
    for system in systems:
        get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

    evaluate_model(model, systems[:5], targets=target_info_dict, is_training=False)
