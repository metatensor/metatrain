import copy
import random

import metatensor.torch
import numpy as np
import torch
from omegaconf import OmegaConf

from metatrain.gap import GAP, Trainer
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.data.readers.ase import read
from metatrain.utils.data.target_info import get_energy_target_info

from . import DATASET_ETHANOL_PATH, DATASET_PATH, DEFAULT_HYPERS


torch.set_default_dtype(torch.float64)  # GAP only supports float64


# reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def test_regression_init():
    """Perform a regression test on the model at initialization"""
    targets = {}
    targets["mtt::U0"] = get_energy_target_info({"unit": "eV"})

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=targets
    )
    GAP(DEFAULT_HYPERS["model"], dataset_info)


def test_regression_train_and_invariance():
    """Perform a regression test on the model when trained for 2 epoch on a small
    dataset.  We perform also the invariance test here because one needs a trained model
    for this.
    """

    systems = read_systems(DATASET_PATH)

    conf = {
        "mtt::U0": {
            "quantity": "energy",
            "read_from": DATASET_PATH,
            "reader": "ase",
            "key": "U0",
            "unit": "kcal/mol",
            "type": "scalar",
            "per_atom": False,
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, _ = read_targets(OmegaConf.create(conf))
    dataset = Dataset.from_dict({"system": systems, "mtt::U0": targets["mtt::U0"]})

    target_info_dict = {}
    target_info_dict["mtt::U0"] = get_energy_target_info({"unit": "eV"})

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=target_info_dict
    )

    gap = GAP(DEFAULT_HYPERS["model"], dataset_info)
    trainer = Trainer(DEFAULT_HYPERS["training"])
    trainer.train(
        model=gap,
        dtype=torch.float64,
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir=".",
    )
    gap.eval()

    # Predict on the first five systems
    output = gap(systems[:5], {"mtt::U0": gap.outputs["mtt::U0"]})

    expected_output = torch.tensor(
        [[-40.5891], [-56.7122], [-76.4146], [-77.3364], [-93.4905]]
    )

    assert torch.allclose(output["mtt::U0"].block().values, expected_output, rtol=0.3)

    # Tests that the model is rotationally invariant
    system = read(DATASET_PATH)
    system.numbers = np.ones(len(system.numbers))

    original_system = copy.deepcopy(system)
    system.rotate(48, "y")

    original_output = gap(
        [metatensor.torch.atomistic.systems_to_torch(original_system)],
        {"mtt::U0": gap.outputs["mtt::U0"]},
    )
    rotated_output = gap(
        [metatensor.torch.atomistic.systems_to_torch(system)],
        {"mtt::U0": gap.outputs["mtt::U0"]},
    )

    assert torch.allclose(
        original_output["mtt::U0"].block().values,
        rotated_output["mtt::U0"].block().values,
    )


def test_ethanol_regression_train_and_invariance():
    """Perform a regression test on the model when trained for 2 epoch on a small
    dataset.  We perform also the invariance test here because one needs a trained model
    for this.
    """

    systems = read_systems(DATASET_ETHANOL_PATH)

    conf = {
        "energy": {
            "quantity": "energy",
            "read_from": DATASET_ETHANOL_PATH,
            "reader": "ase",
            "key": "energy",
            "type": "scalar",
            "per_atom": False,
            "num_subtargets": 1,
            "forces": {
                "read_from": DATASET_ETHANOL_PATH,
                "reader": "ase",
                "key": "forces",
            },
            "unit": "kcal/mol",
            "stress": False,
            "virial": False,
        }
    }

    targets, _ = read_targets(OmegaConf.create(conf))
    dataset = Dataset.from_dict({"system": systems, "energy": targets["energy"]})

    hypers = copy.deepcopy(DEFAULT_HYPERS)
    hypers["model"]["krr"]["num_sparse_points"] = 900

    target_info_dict = {
        "energy": get_energy_target_info({"unit": "eV"}, add_position_gradients=True)
    }

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=target_info_dict
    )

    gap = GAP(hypers["model"], dataset_info)
    trainer = Trainer(hypers["training"])
    trainer.train(
        model=gap,
        dtype=torch.float64,
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir=".",
    )
    gap.eval()

    # Predict on the first five systems
    output = gap(systems[:5], {"energy": gap.outputs["energy"]})
    data = read(DATASET_ETHANOL_PATH, ":5", format="extxyz")

    expected_output = torch.tensor([[i.info["energy"]] for i in data])
    assert torch.allclose(output["energy"].block().values, expected_output, rtol=0.1)

    # TODO: check accuracy of training forces
    # expected_forces = torch.vstack([torch.Tensor(i.arrays["forces"]) for i in data])

    # Tests that the model is rotationally invariant
    system = read(DATASET_ETHANOL_PATH)

    original_system = copy.deepcopy(system)
    system.rotate(48, "y")

    original_output = gap(
        [metatensor.torch.atomistic.systems_to_torch(original_system)],
        {"energy": gap.outputs["energy"]},
    )
    rotated_output = gap(
        [metatensor.torch.atomistic.systems_to_torch(system)],
        {"energy": gap.outputs["energy"]},
    )

    assert torch.allclose(
        original_output["energy"].block().values,
        rotated_output["energy"].block().values,
    )
