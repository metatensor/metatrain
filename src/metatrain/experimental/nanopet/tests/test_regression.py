import os
import random

import numpy as np
import torch
from metatomic.torch import ModelOutput
from omegaconf import OmegaConf

from metatrain.experimental.nanopet import NanoPET, Trainer
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
from metatrain.utils.omegaconf import CONF_LOSS

from . import DATASET_PATH, DEFAULT_HYPERS, MODEL_HYPERS


def test_regression_init():
    """Perform a regression test on the model at initialization"""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    os.environ["PYTHONHASHSEED"] = str(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    targets = {}
    targets["mtt::U0"] = get_energy_target_info({"quantity": "energy", "unit": "eV"})

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=targets
    )
    model = NanoPET(MODEL_HYPERS, dataset_info)

    # Predict on the first five systems
    systems = read_systems(DATASET_PATH)[:5]
    systems = [system.to(torch.float32) for system in systems]
    for system in systems:
        get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

    output = model(
        systems,
        {"mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=False)},
    )

    expected_output = torch.tensor(
        [
            [0.163995444775],
            [0.068577021360],
            [-0.003538529389],
            [0.049175731838],
            [0.044154867530],
        ]
    )

    # if you need to change the hardcoded values:
    # torch.set_printoptions(precision=12)
    # print(output["mtt::U0"].block().values)

    torch.testing.assert_close(output["mtt::U0"].block().values, expected_output)


def test_regression_train():
    """Regression test for the model when trained for 2 epoch on a small dataset"""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    os.environ["PYTHONHASHSEED"] = str(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    systems = read_systems(DATASET_PATH)

    conf = {
        "mtt::U0": {
            "quantity": "energy",
            "read_from": DATASET_PATH,
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "per_atom": False,
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, target_info_dict = read_targets(OmegaConf.create(conf))
    dataset = Dataset.from_dict({"system": systems, "mtt::U0": targets["mtt::U0"]})

    hypers = DEFAULT_HYPERS.copy()
    hypers["training"]["num_epochs"] = 2
    loss_conf = OmegaConf.create({"mtt::U0": CONF_LOSS.copy()})
    OmegaConf.resolve(loss_conf)
    hypers["training"]["loss"] = loss_conf

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=target_info_dict
    )
    model = NanoPET(MODEL_HYPERS, dataset_info)

    hypers["training"]["num_epochs"] = 1
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
    output = model(
        systems[:5],
        {"mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=False)},
    )

    expected_output = torch.tensor(
        [
            [0.260419785976],
            [0.258879989386],
            [0.127974838018],
            [0.177975922823],
            [0.136635094881],
        ]
    )

    # if you need to change the hardcoded values:
    # torch.set_printoptions(precision=12)
    # print(output["mtt::U0"].block().values)

    torch.testing.assert_close(output["mtt::U0"].block().values, expected_output)
