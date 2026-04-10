import random

import numpy as np
import torch
from metatomic.torch import ModelOutput
from omegaconf import OmegaConf

from metatrain.experimental.dpa3 import DPA3, Trainer
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import (
    read_systems,
    read_targets,
)
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DATASET_PATH, DATASET_WITH_FORCES_PATH, DEFAULT_HYPERS, MODEL_HYPERS


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
    model = DPA3(MODEL_HYPERS, dataset_info).to("cpu")

    # Predict on the first five systems
    systems = read_systems(DATASET_PATH)[:5]
    systems = [system.to(torch.float64) for system in systems]
    for system in systems:
        get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

    output = model(
        systems,
        {"mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=False)},
    )

    expected_output = torch.tensor(
        [
            [8.893970727921],
            [7.150644659996],
            [5.338875532150],
            [7.145487308502],
            [5.402073264122],
        ],
        dtype=torch.float64,
    )

    # if you need to change the hardcoded values:
    torch.set_printoptions(precision=12)
    print(output["mtt::U0"].block().values)

    torch.testing.assert_close(output["mtt::U0"].block().values, expected_output)


def test_regression_energies_forces_train():
    """Regression test for the model when trained for 2 epoch on a small dataset"""
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
    hypers["training"]["num_epochs"] = 1
    hypers["training"]["scheduler_patience"] = 1
    hypers["training"]["fixed_composition_weights"] = {}
    loss_conf = OmegaConf.create({"energy": init_with_defaults(LossSpecification)})
    hypers["training"]["loss"] = loss_conf

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[6], targets=target_info_dict
    )
    model = DPA3(MODEL_HYPERS, dataset_info).to("cpu")
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
    systems = [system.to(torch.float64) for system in systems]
    for system in systems:
        get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

    output = evaluate_model(
        model, systems[:5], targets=target_info_dict, is_training=False
    )

    expected_output = torch.tensor(
        [
            [1.159545063972],
            [1.186555385590],
            [1.196010828018],
            [1.124937891960],
            [1.161755204201],
        ],
        dtype=torch.float64,
    )

    expected_gradients_output = torch.tensor(
        [7.167162038968e-06, -6.548931468501e-03, 6.634737014065e-02],
        dtype=torch.float64,
    )

    # if you need to change the hardcoded values:
    torch.set_printoptions(precision=12)
    print(output["energy"].block().values)
    print(output["energy"].block().gradient("positions").values.squeeze(-1)[0])

    # Training is done in float32, so float64-default atol=1e-7 is too tight.
    torch.testing.assert_close(
        output["energy"].block().values, expected_output, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        output["energy"].block().gradient("positions").values[0, :, 0],
        expected_gradients_output,
        atol=1e-5,
        rtol=1e-5,
    )
