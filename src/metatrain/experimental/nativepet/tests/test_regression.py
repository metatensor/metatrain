import random

import numpy as np
import torch
from metatensor.torch.atomistic import ModelOutput
from omegaconf import OmegaConf

from metatrain.experimental.nativepet import NativePET, Trainer
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import (
    read_systems,
    read_targets,
)
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DATASET_PATH, DATASET_WITH_FORCES_PATH, DEFAULT_HYPERS, MODEL_HYPERS


# reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def test_regression_init():
    """Perform a regression test on the model at initialization"""

    targets = {}
    targets["mtt::U0"] = get_energy_target_info({"quantity": "energy", "unit": "eV"})

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=targets
    )
    model = NativePET(MODEL_HYPERS, dataset_info)

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
            [1.645375013351],
            [1.410712957382],
            [0.570396900177],
            [0.294374406338],
            [0.099420711398],
        ]
    )

    # if you need to change the hardcoded values:
    torch.set_printoptions(precision=12)
    print(output["mtt::U0"].block().values)

    torch.testing.assert_close(output["mtt::U0"].block().values, expected_output)


def test_regression_energies_train():
    """Perform a regression test on the model when
    trained for 2 epoch on a small dataset, energies only"""

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
    hypers["training"]["scheduler_patience"] = 1

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=target_info_dict
    )
    model = NativePET(MODEL_HYPERS, dataset_info)

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
            [-0.527752280235],
            [-0.307992666960],
            [-0.278871238232],
            [-0.120195508003],
            [0.147994175553],
        ]
    )

    # if you need to change the hardcoded values:
    torch.set_printoptions(precision=12)
    print(output["mtt::U0"].block().values)

    torch.testing.assert_close(output["mtt::U0"].block().values, expected_output)


def test_regression_energies_forces_train():
    """Perform a regression test on the model when
    trained for 2 epoch on a small dataset with energies
    and forces"""

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
    model = NativePET(MODEL_HYPERS, dataset_info)
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

    output = evaluate_model(
        model, systems[:5], targets=target_info_dict, is_training=False
    )

    expected_output = torch.tensor(
        [
            [25.142356872559],
            [25.295734405518],
            [25.210765838623],
            [25.256343841553],
            [25.199890136719],
        ]
    )

    expected_gradients_output = torch.tensor(
        [0.204146623611, -0.101296871901, -0.270830482244]
    )

    # if you need to change the hardcoded values:
    torch.set_printoptions(precision=12)
    print(output["energy"].block().values)
    print(output["energy"].block().gradient("positions").values.squeeze(-1)[0])

    torch.testing.assert_close(output["energy"].block().values, expected_output)
    torch.testing.assert_close(
        output["energy"].block().gradient("positions").values[0, :, 0],
        expected_gradients_output,
    )
