import copy
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
    targets["mtt::U0"] = get_energy_target_info("mtt::U0", {"unit": "eV"})

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
    hypers["training"]["num_workers"] = 0  # for reproducibility

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
    systems = [system.to(torch.float32, device) for system in systems]
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
            [0.783539175987],
            [0.388317406178],
            [1.728710055351],
            [6.365263938904],
            [0.569676578045],
        ],
        device=device,
    )

    # if you need to change the hardcoded values:
    # torch.set_printoptions(precision=12)
    # print(output["mtt::U0"].block().values)

    torch.testing.assert_close(
        output["mtt::U0"].block().values, expected_output, rtol=5e-5, atol=1e-5
    )


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

    hypers = copy.deepcopy(DEFAULT_HYPERS)
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
    hypers["training"]["num_workers"] = 0  # for reproducibility
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
    systems = [system.to(torch.float32, device) for system in systems]
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
                -0.015746125951,
                0.008406482637,
                0.010487615131,
                0.011963039637,
                0.028127808124,
                -0.009492993355,
                0.021495090798,
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
                0.011813754216,
                0.002038915642,
                -0.004326812923,
                -0.026226855814,
                -0.015365801752,
                -0.006548378151,
                -0.008180803619,
            ],
        ],
        device=device,
    )

    # if you need to change the hardcoded values:
    # torch.set_printoptions(precision=12)
    # print(output["mtt::electron_density_basis"][1].values[2])

    torch.testing.assert_close(
        output["mtt::electron_density_basis"][1].values[2], expected_output
    )
