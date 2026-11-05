"""Training smoke test on the carbon subset.

Numerical snapshots are not pinned: deepmd-kit draws the initial weights, and
those values move when the library does. The test checks that one epoch of
training produces finite energies and forces.
"""

import random

import numpy as np
import torch
from omegaconf import OmegaConf

from metatrain.experimental.se_e2_a import Trainer
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DATASET_WITH_FORCES_PATH, DEFAULT_HYPERS
from .test_basic import _minimal_hypers


def test_one_epoch_produces_finite_energy_and_forces():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    systems = read_systems(DATASET_WITH_FORCES_PATH)[:8]
    conf = {
        "energy": {
            "quantity": "energy",
            "read_from": DATASET_WITH_FORCES_PATH,
            "reader": "ase",
            "key": "energy",
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": {"read_from": DATASET_WITH_FORCES_PATH, "key": "force"},
            "stress": False,
            "virial": False,
        }
    }
    targets, target_info_dict = read_targets(OmegaConf.create(conf))
    dataset = Dataset.from_dict(
        {"system": systems, "energy": targets["energy"][: len(systems)]}
    )
    hypers = DEFAULT_HYPERS.copy()
    hypers["training"]["num_epochs"] = 1
    hypers["training"]["batch_size"] = 2
    hypers["training"]["scheduler_patience"] = 5
    hypers["training"]["fixed_composition_weights"] = {}
    hypers["training"]["loss"] = OmegaConf.create(
        {"energy": init_with_defaults(LossSpecification)}
    )
    model_hypers = _minimal_hypers("experimental.se_e2_a")
    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[6], targets=target_info_dict
    )
    trainer = Trainer(hypers["training"])
    model = trainer.setup(model_hypers, dataset_info).to("cpu")
    trainer.train(
        model=model,
        dtype=torch.float32,
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir=".",
    )

    evaluated = [system.to(torch.float32) for system in systems[:4]]
    for system in evaluated:
        get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    output = evaluate_model(
        model, evaluated, targets=target_info_dict, is_training=False
    )
    energy = output["energy"].block().values
    forces = output["energy"].block().gradient("positions").values
    assert energy.shape == (len(evaluated), 1)
    assert torch.isfinite(energy).all()
    n_atoms = sum(len(system) for system in evaluated)
    assert forces.shape[0] == n_atoms
    assert torch.isfinite(forces).all()
