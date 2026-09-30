"""Pinned outputs so later changes to LOREM are easy to spot.

Same idea as ``pet/tests/test_regression.py``: fixed seed, a short model,
and hardcoded predictions. The model is smaller than the paper defaults so
the test stays cheap; the numbers still move if the forward pass or the
trainer changes.
"""

import copy
import random

import numpy as np
import pytest
import torch
from metatomic.torch import ModelOutput
from omegaconf import OmegaConf

from metatrain.experimental.lorem import LOREM, Trainer
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DATASET_PATH, DATASET_WITH_FORCES_PATH, DEFAULT_HYPERS, MODEL_HYPERS


pytest.importorskip("torchpme")


def _regression_hypers() -> dict:
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["cutoff"] = 3.0
    hypers["max_degree"] = 1
    hypers["max_degree_lr"] = 0
    hypers["num_features"] = 8
    hypers["num_spherical_features"] = 2
    hypers["num_radial"] = 4
    hypers["num_message_passing"] = 0
    return hypers


def test_regression_init():
    """Regression test for the model at initialization."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    targets = {
        "mtt::U0": get_energy_target_info(
            "mtt::U0", {"quantity": "energy", "unit": "eV"}
        )
    }
    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=targets
    )
    model = LOREM(_regression_hypers(), dataset_info)

    systems = read_systems(DATASET_PATH)[:5]
    systems = [system.to(torch.float32) for system in systems]
    for system in systems:
        get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

    output = model(
        systems,
        {"mtt::U0": ModelOutput(unit="", sample_kind="system")},
    )

    expected_output = torch.tensor(
        [
            [0.625304460526],
            [0.490911960602],
            [0.351827710867],
            [0.555834591389],
            [0.413867086172],
        ]
    )

    # if you need to change the hardcoded values:
    # torch.set_printoptions(precision=12)
    # print(output["mtt::U0"].block().values)

    torch.testing.assert_close(output["mtt::U0"].block().values, expected_output)


def test_regression_energies_forces_train(tmp_path, monkeypatch):
    """Regression test for two epochs on a small dataset with forces."""
    monkeypatch.chdir(tmp_path)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    systems = read_systems(DATASET_WITH_FORCES_PATH)

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
    dataset = Dataset.from_dict({"system": systems, "energy": targets["energy"]})

    hypers = copy.deepcopy(DEFAULT_HYPERS)
    hypers["training"]["num_epochs"] = 2
    hypers["training"]["batch_size"] = 4
    hypers["training"]["scheduler_patience"] = 1
    hypers["training"]["atomic_baseline"] = {}
    loss_conf = {"energy": init_with_defaults(LossSpecification)}
    loss_conf["energy"]["gradients"] = {
        "positions": init_with_defaults(LossSpecification)
    }
    loss_conf = OmegaConf.create(loss_conf)
    OmegaConf.resolve(loss_conf)
    hypers["training"]["loss"] = loss_conf

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[6], targets=target_info_dict
    )
    model = LOREM(_regression_hypers(), dataset_info)
    trainer = Trainer(hypers["training"])
    trainer.train(
        model=model,
        dtype=torch.float32,
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir=".",
    )

    systems = [system.to(torch.float32) for system in systems]
    for system in systems:
        get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

    output = evaluate_model(
        model, systems[:5], targets=target_info_dict, is_training=False
    )

    expected_output = torch.tensor(
        [
            [0.026406854391],
            [0.040477454662],
            [0.036829620600],
            [0.023532927036],
            [0.046602159739],
        ]
    )
    expected_gradients_output = torch.tensor(
        [-0.000752118300, 0.002826320706, -0.001288130414]
    )

    # if you need to change the hardcoded values:
    # torch.set_printoptions(precision=12)
    # print(output["energy"].block().values)
    # print(output["energy"].block().gradient("positions").values.squeeze(-1)[0])

    torch.testing.assert_close(output["energy"].block().values, expected_output)
    torch.testing.assert_close(
        output["energy"].block().gradient("positions").values[0, :, 0],
        expected_gradients_output,
    )
