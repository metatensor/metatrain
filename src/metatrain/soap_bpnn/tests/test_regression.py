import random

import numpy as np
import pytest
import torch
from metatomic.torch import ModelOutput
from omegaconf import OmegaConf

from metatrain.soap_bpnn import SoapBpnn, Trainer
from metatrain.utils.data import Dataset, DatasetInfo, get_dataset
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)
from metatrain.utils.omegaconf import CONF_LOSS

from . import DATASET_PATH, DEFAULT_HYPERS, MODEL_HYPERS, SPHERICAL_DISK_DATASET_PATH


def test_regression_init():
    """Perform a regression test on the model at initialization"""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    targets = {}
    targets["mtt::U0"] = get_energy_target_info({"unit": "eV"})

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=targets
    )
    model = SoapBpnn(MODEL_HYPERS, dataset_info)
    requested_neighbor_lists = get_requested_neighbor_lists(model)

    # Predict on the first five systems
    systems = read_systems(DATASET_PATH)[:5]
    systems = [system.to(torch.float32) for system in systems]
    systems = [
        get_system_with_neighbor_lists(system, requested_neighbor_lists)
        for system in systems
    ]

    output = model(
        systems,
        {"mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=False)},
    )

    expected_output = torch.tensor(
        [
            [0.115978844464],
            [0.074449732900],
            [-0.024028975517],
            [0.192573457956],
            [-0.221303701401],
        ]
    )

    # if you need to change the hardcoded values:
    # torch.set_printoptions(precision=12)
    # print(output["mtt::U0"].block().values)

    torch.testing.assert_close(output["mtt::U0"].block().values, expected_output)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_regression_train(device):
    """Regression test for the model when trained for 2 epoch on a small dataset"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

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
    model = SoapBpnn(MODEL_HYPERS, dataset_info)
    requested_neighbor_lists = get_requested_neighbor_lists(model)

    hypers["training"]["num_epochs"] = 1
    trainer = Trainer(hypers["training"])
    trainer.train(
        model=model,
        dtype=torch.float32,
        devices=[torch.device(device)],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir=".",
    )

    # Predict on the first five systems
    systems = [system.to(torch.float32) for system in systems]
    systems = [
        get_system_with_neighbor_lists(system, requested_neighbor_lists)
        for system in systems
    ]
    output = model(
        systems[:5],
        {"mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=False)},
    )

    expected_output = torch.tensor(
        [
            [1.313831329346],
            [4.282802581787],
            [5.629218101501],
            [4.297019958496],
            [2.226531982422],
        ]
    )

    # if you need to change the hardcoded values:
    # torch.set_printoptions(precision=12)
    # print(output["mtt::U0"].block().values)

    torch.testing.assert_close(output["mtt::U0"].block().values, expected_output)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_regression_train_spherical(device):
    """Regression test for the model when trained for 2 epoch on a small dataset"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    conf = {
        "systems": {"read_from": SPHERICAL_DISK_DATASET_PATH},
        "targets": {
            "mtt::electron_density_basis": {
                "quantity": "",
                "unit": "",
                "read_from": SPHERICAL_DISK_DATASET_PATH,
                "type": {
                    "spherical": {
                        "irreps": [
                            {"o3_lambda": 0, "o3_sigma": 1},
                            {"o3_lambda": 1, "o3_sigma": 1},
                            {"o3_lambda": 2, "o3_sigma": 1},
                            {"o3_lambda": 3, "o3_sigma": 1},
                        ]
                    },
                },
                "per_atom": True,
                "num_subtargets": 1,  # dummy value
            },
        },
    }

    dataset, target_info_dict, _ = get_dataset(conf)

    hypers = DEFAULT_HYPERS.copy()
    hypers["training"]["num_epochs"] = 2
    hypers["training"]["batch_size"] = 1
    hypers["training"]["loss"]["mtt::electron_density_basis"] = hypers["training"][
        "loss"
    ].pop("mtt::U0")

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=target_info_dict
    )
    model = SoapBpnn(MODEL_HYPERS, dataset_info)
    requested_neighbor_lists = get_requested_neighbor_lists(model)

    hypers["training"]["num_epochs"] = 1
    trainer = Trainer(hypers["training"])
    trainer.train(
        model=model,
        dtype=torch.float32,
        devices=[torch.device(device)],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir=".",
    )

    # Predict on the first five systems
    systems = [sample["system"] for sample in dataset]
    systems = [system.to(torch.float32) for system in systems]
    systems = [
        get_system_with_neighbor_lists(system, requested_neighbor_lists)
        for system in systems
    ]
    output = model(
        systems,
        {
            "mtt::electron_density_basis": ModelOutput(
                quantity="", unit="", per_atom=True
            )
        },
    )

    expected_output = torch.tensor(
        [
            [
                -0.057801067829,
                -0.022922841832,
                -0.013157465495,
                0.003876133356,
                0.008559819311,
                0.039749406278,
                0.013140974566,
            ],
            [
                0.000000000000,
                0.000000000000,
                0.000000000000,
                0.000000000000,
                0.000000000000,
                0.000000000000,
                0.000000000000,
            ],
            [
                0.020274644718,
                0.005657686852,
                0.001842519501,
                -0.014781145379,
                0.003011089284,
                -0.011008090340,
                -0.012492593378,
            ],
        ],
    )

    # if you need to change the hardcoded values:
    # torch.set_printoptions(precision=12)
    # print(output["mtt::electron_density_basis"][1].values[2])

    torch.testing.assert_close(
        output["mtt::electron_density_basis"][1].values[2], expected_output
    )
