import copy
import random

import ase.io
import metatensor.torch
import numpy as np
import torch
from omegaconf import OmegaConf

from metatensor.models.experimental.gap import GAP, Trainer
from metatensor.models.utils.architectures import get_default_hypers
from metatensor.models.utils.data import Dataset, DatasetInfo, TargetInfo
from metatensor.models.utils.data.readers import read_systems, read_targets

from . import DATASET_ETHANOL_PATH, DATASET_PATH


DEFAULT_HYPERS = get_default_hypers("experimental.gap")


torch.set_default_dtype(torch.float64)  # GAP only supports float64


# reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def test_regression_init():
    """Perform a regression test on the model at initialization"""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "mtm::U0": TargetInfo(
                quantity="energy",
                unit="eV",
            ),
        },
    )
    GAP(DEFAULT_HYPERS["model"], dataset_info)


def test_regression_train_and_invariance():
    """Perform a regression test on the model when trained for 2 epoch on a small
    dataset.  We perform also the invariance test here because one needs a trained model
    for this.
    """

    systems = read_systems(DATASET_PATH, dtype=torch.float64)

    conf = {
        "mtm::U0": {
            "quantity": "energy",
            "read_from": DATASET_PATH,
            "file_format": ".xyz",
            "key": "U0",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets = read_targets(OmegaConf.create(conf), dtype=torch.float64)
    dataset = Dataset({"system": systems, "mtm::U0": targets["mtm::U0"]})

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "mtm::U0": TargetInfo(
                quantity="energy",
                unit="eV",
            ),
        },
    )
    gap = GAP(DEFAULT_HYPERS["model"], dataset_info)
    trainer = Trainer(DEFAULT_HYPERS["training"])
    trainer.train(gap, [torch.device("cpu")], [dataset], [dataset], ".")

    # Predict on the first five systems
    output = gap(systems[:5], {"mtm::U0": gap.outputs["mtm::U0"]})

    expected_output = torch.tensor(
        [[-40.5891], [-56.7122], [-76.4146], [-77.3364], [-93.4905]]
    )

    assert torch.allclose(output["mtm::U0"].block().values, expected_output, rtol=0.3)

    # Tests that the model is rotationally invariant
    system = ase.io.read(DATASET_PATH)
    system.numbers = np.ones(len(system.numbers))

    original_system = copy.deepcopy(system)
    system.rotate(48, "y")

    original_output = gap(
        [metatensor.torch.atomistic.systems_to_torch(original_system)],
        {"mtm::U0": gap.outputs["mtm::U0"]},
    )
    rotated_output = gap(
        [metatensor.torch.atomistic.systems_to_torch(system)],
        {"mtm::U0": gap.outputs["mtm::U0"]},
    )

    assert torch.allclose(
        original_output["mtm::U0"].block().values,
        rotated_output["mtm::U0"].block().values,
    )


def test_ethanol_regression_train_and_invariance():
    """Perform a regression test on the model when trained for 2 epoch on a small
    dataset.  We perform also the invariance test here because one needs a trained model
    for this.
    """

    systems = read_systems(DATASET_ETHANOL_PATH, dtype=torch.float64)

    conf = {
        "energy": {
            "quantity": "energy",
            "read_from": DATASET_ETHANOL_PATH,
            "file_format": ".xyz",
            "key": "energy",
            "forces": {
                "read_from": DATASET_ETHANOL_PATH,
                "file_format": ".xyz",
                "key": "forces",
            },
            "stress": False,
            "virial": False,
        }
    }

    targets = read_targets(OmegaConf.create(conf), dtype=torch.float64)
    dataset = Dataset({"system": systems, "energy": targets["energy"]})

    hypers = copy.deepcopy(DEFAULT_HYPERS)
    hypers["model"]["krr"]["num_sparse_points"] = 900

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": TargetInfo(
                quantity="energy",
                unit="kcal/mol",
            ),
        },
    )

    gap = GAP(hypers["model"], dataset_info)
    trainer = Trainer(hypers["training"])
    trainer.train(gap, [torch.device("cpu")], [dataset], [dataset], ".")

    # Predict on the first five systems
    output = gap(systems[:5], {"energy": gap.outputs["energy"]})
    data = ase.io.read(DATASET_ETHANOL_PATH, ":5", format="extxyz")

    expected_output = torch.tensor([[i.info["energy"]] for i in data])
    assert torch.allclose(output["energy"].block().values, expected_output, rtol=0.1)

    # TODO: check accuracy of training forces
    # expected_forces = torch.vstack([torch.Tensor(i.arrays["forces"]) for i in data])

    # Tests that the model is rotationally invariant
    system = ase.io.read(DATASET_ETHANOL_PATH)

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
