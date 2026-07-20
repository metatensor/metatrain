import copy
import random

import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System
from omegaconf import OmegaConf

from metatrain.pet import PET, Trainer
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import Dataset, DatasetInfo, get_dataset
from metatrain.utils.data.readers import (
    read_systems,
    read_targets,
)
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from . import (
    DATASET_PATH,
    DATASET_WITH_FORCES_PATH,
    DEFAULT_HYPERS,
    MODEL_HYPERS,
    SPHERICAL_DISK_DATASET_PATH,
)


def test_regression_init():
    """Regression test for the model at initialization"""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    targets = {}
    targets["mtt::U0"] = get_energy_target_info(
        "mtt::U0", {"quantity": "energy", "unit": "eV"}
    )

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=targets
    )
    model = PET(MODEL_HYPERS, dataset_info)

    # Predict on the first five systems
    systems = read_systems(DATASET_PATH)[:5]
    systems = [system.to(torch.float32) for system in systems]
    for system in systems:
        get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

    output = model(
        systems,
        {"mtt::U0": ModelOutput(quantity="energy", unit="", sample_kind="system")},
    )

    expected_output = torch.tensor(
        [
            [1.146098375320],
            [0.171331465244],
            [0.539504408836],
            [0.861489117146],
            [0.177449733019],
        ]
    )

    # if you need to change the hardcoded values:
    # torch.set_printoptions(precision=12)
    # print(output["mtt::U0"].block().values)

    torch.testing.assert_close(output["mtt::U0"].block().values, expected_output)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_regression_energies_forces_train(device):
    """Regression test for the model when trained for 2 epoch on a small dataset"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

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
    targets = {"energy": targets["energy"]}
    dataset = Dataset.from_dict({"system": systems, "energy": targets["energy"]})
    hypers = copy.deepcopy(DEFAULT_HYPERS)
    hypers["training"]["num_epochs"] = 2
    hypers["training"]["num_workers"] = 0  # for reproducibility
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
    model = PET(MODEL_HYPERS, dataset_info)
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
    for system in systems:
        get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

    output = evaluate_model(
        model, systems[:5], targets=target_info_dict, is_training=False
    )

    expected_output = torch.tensor(
        [
            [2.834107398987],
            [2.889175415039],
            [2.844632148743],
            [2.930275917053],
            [2.905971050262],
        ],
        device=device,
    )

    expected_gradients_output = torch.tensor(
        [0.010308556259, 0.041110992432, -0.090265549719], device=device
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


@pytest.mark.parametrize("batch_size", [1, 2])
def test_regression_energy_non_conservative_stress(batch_size):
    """Regression test for PET setup with mixed energy and NC stress targets."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    systems = [
        System(
            positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            types=torch.tensor([8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
        System(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64
            ),
            types=torch.tensor([8, 8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
        System(
            positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            types=torch.tensor([8]),
            cell=torch.zeros(3, 3, dtype=torch.float64),
            pbc=torch.tensor([False, False, False]),
        ),
    ]
    energies = [1.0, 8.0, 2.0]
    valid_stresses = [
        torch.full((1, 3, 3, 1), 2.0, dtype=torch.float64),
        torch.full((1, 3, 3, 1), 4.0, dtype=torch.float64),
    ]
    nan_stress = torch.full((1, 3, 3, 1), torch.nan, dtype=torch.float64)
    stresses = [*valid_stresses, nan_stress]
    energy_targets = [
        TensorMap(
            keys=Labels.single(),
            blocks=[
                TensorBlock(
                    values=torch.tensor([[energy]], dtype=torch.float64),
                    samples=Labels(names=["system"], values=torch.tensor([[i]])),
                    components=[],
                    properties=Labels.range("energy", 1),
                )
            ],
        )
        for i, energy in enumerate(energies)
    ]
    targets = [
        TensorMap(
            keys=Labels.single(),
            blocks=[
                TensorBlock(
                    values=stress,
                    samples=Labels(names=["system"], values=torch.tensor([[i]])),
                    components=[Labels.range("xyz_1", 3), Labels.range("xyz_2", 3)],
                    properties=Labels.range("non_conservative_stress", 1),
                )
            ],
        )
        for i, stress in enumerate(stresses)
    ]
    dataset = Dataset.from_dict(
        {
            "system": systems,
            "energy": energy_targets,
            "non_conservative_stress": targets,
        }
    )

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[8],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            ),
            "non_conservative_stress": get_generic_target_info(
                "non_conservative_stress",
                {
                    "quantity": "stress",
                    "unit": "eV/A^3",
                    "type": {"cartesian": {"rank": 2}},
                    "sample_kind": "system",
                    "num_subtargets": 1,
                },
            ),
        },
    )

    hypers = copy.deepcopy(DEFAULT_HYPERS)
    hypers["training"]["batch_size"] = batch_size
    hypers["training"]["num_epochs"] = 0
    hypers["training"]["num_workers"] = 0
    hypers["training"]["atomic_baseline"] = {"energy": 0.0}
    hypers["training"]["per_structure_targets"] = ["non_conservative_stress"]
    loss_conf = OmegaConf.create(
        {
            "energy": init_with_defaults(LossSpecification),
            "non_conservative_stress": init_with_defaults(LossSpecification),
        }
    )
    OmegaConf.resolve(loss_conf)
    hypers["training"]["loss"] = loss_conf

    model_hypers = copy.deepcopy(MODEL_HYPERS)
    model_hypers["zbl"] = False
    model = PET(model_hypers, dataset_info)
    trainer = Trainer(hypers["training"])
    # num_epochs=0 keeps this test focused on pre-training scaler fitting.
    trainer.train(
        model=model,
        dtype=torch.float32,
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir=".",
    )

    energy_scale = model.scaler.model.scales["energy"].block().values[0, 0]
    stress_scale = (
        model.scaler.model.scales["non_conservative_stress"].block().values[0, 0]
    )

    eval_systems = [system.to(torch.float32) for system in systems]
    for system in eval_systems:
        get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

    outputs = model(
        eval_systems,
        {
            "energy": ModelOutput(quantity="energy", unit="", sample_kind="system"),
            "non_conservative_stress": ModelOutput(sample_kind="system"),
        },
    )
    energy_output = outputs["energy"].block().values
    stress_output = outputs["non_conservative_stress"].block().values

    expected_energy_scale = torch.tensor(2.6457513110645907, dtype=torch.float64)
    expected_stress_scale = torch.tensor(3.1622776601683795, dtype=torch.float64)
    expected_energy_output = torch.tensor(
        [[40.56496047973633], [162.2603759765625], [0.08803563565015793]],
        dtype=torch.float32,
    )
    expected_stress_output = torch.tensor(
        [
            [
                [[-37.251258850097656], [10.948074340820312], [34.96856689453125]],
                [[10.948074340820312], [13.099261283874512], [8.543399810791016]],
                [[34.96856689453125], [8.543399810791016], [-18.625041961669922]],
            ],
            [
                [[-149.29901123046875], [43.95909881591797], [140.06430053710938]],
                [[43.95909881591797], [52.38413619995117], [34.086326599121094]],
                [[140.06430053710938], [34.086326599121094], [-74.72312927246094]],
            ],
            [
                [[0.0], [0.0], [0.0]],
                [[0.0], [0.0], [0.0]],
                [[0.0], [0.0], [0.0]],
            ],
        ],
        dtype=torch.float32,
    )

    # if you need to change the hardcoded values:
    # torch.set_printoptions(precision=12)
    # print(repr(energy_scale.detach().cpu()))
    # print(repr(stress_scale.detach().cpu()))
    # print(repr(energy_output.detach().cpu()))
    # print(repr(stress_output.detach().cpu()))

    torch.testing.assert_close(energy_scale, expected_energy_scale)
    torch.testing.assert_close(stress_scale, expected_stress_scale)
    torch.testing.assert_close(energy_output, expected_energy_output)
    torch.testing.assert_close(stress_output, expected_stress_output)


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
                "sample_kind": "atom",
                "num_subtargets": 1,  # dummy value
            },
        },
    }

    dataset, target_info_dict, extra_data_info = get_dataset(conf)

    hypers = get_default_hypers("pet")
    hypers["training"]["num_epochs"] = 2
    hypers["training"]["batch_size"] = 1
    loss_conf = {"mtt::electron_density_basis": init_with_defaults(LossSpecification)}
    loss_conf = OmegaConf.create(loss_conf)
    OmegaConf.resolve(loss_conf)
    hypers["training"]["loss"] = loss_conf

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets=target_info_dict,
        extra_data=extra_data_info,
    )
    model = PET(MODEL_HYPERS, dataset_info)
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
                quantity="", unit="", sample_kind="atom"
            )
        },
    )

    expected_output = torch.tensor(
        [
            [
                0.103833243251,
                0.069846123457,
                0.167408689857,
                0.048254013062,
                -0.124650284648,
                0.303171575069,
                -0.168893411756,
            ],
            [
                -0.362822860479,
                -0.363558739424,
                0.017610874027,
                0.078924819827,
                0.066195741296,
                0.115145877004,
                0.173804253340,
            ],
            [
                -0.064839750528,
                -0.213863357902,
                0.215947464108,
                0.113143950701,
                -0.066840000451,
                -0.264434903860,
                0.057084750384,
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
