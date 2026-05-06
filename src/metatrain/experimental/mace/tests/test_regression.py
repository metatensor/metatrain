import copy
import random

import numpy as np
import pytest
import torch
from metatomic.torch import ModelOutput
from omegaconf import OmegaConf

from metatrain.experimental.mace import MetaMACE, Trainer
from metatrain.utils.data import DatasetInfo, get_dataset
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from . import DEFAULT_HYPERS, MODEL_HYPERS, SPHERICAL_DISK_DATASET_PATH


# reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


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
    loss_conf = {"mtt::electron_density_basis": init_with_defaults(LossSpecification)}
    loss_conf = OmegaConf.create(loss_conf)
    OmegaConf.resolve(loss_conf)
    hypers["training"]["loss"] = loss_conf

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=target_info_dict
    )
    model_hypers = copy.deepcopy(MODEL_HYPERS)
    model = MetaMACE(model_hypers, dataset_info)
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
                -2.441034555435,
                -2.424135446548,
                -0.818219780922,
                1.323156833649,
                -0.593044102192,
                0.973258316517,
                0.263906627893,
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
                0.229800567031,
                0.367431551218,
                0.190992608666,
                -0.106305785477,
                -0.097241364419,
                -0.355860084295,
                -0.122918508947,
            ],
        ],
        device=device,
    )

    # # if you need to change the hardcoded values:
    # torch.set_printoptions(precision=12)
    # print(output["mtt::electron_density_basis"][1].values[2])

    torch.testing.assert_close(
        output["mtt::electron_density_basis"][1].values[2], expected_output
    )
