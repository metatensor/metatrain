import copy
import random

import ase.io
import numpy as np
import rascaline.torch
import torch
from metatensor.learn.data import Dataset
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput
from omegaconf import OmegaConf

from metatensor.models.experimental.gap import DEFAULT_HYPERS, Model, train
from metatensor.models.utils.data import DatasetInfo, TargetInfo
from metatensor.models.utils.data.readers import read_systems, read_targets

from . import DATASET_ETHANOL_PATH, DATASET_PATH


# from pathlib import Path


# reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def test_regression_init():
    """Perform a regression test on the model at initialization"""

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        outputs={
            "U0": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
    )
    Model(capabilities, DEFAULT_HYPERS["model"]).to(torch.float64)


def test_regression_train_and_invariance():
    """Perform a regression test on the model when trained for 2 epoch on a small
    dataset.  We perform also the invariance test here because one needs a trained model
    for this.
    """

    systems = read_systems(DATASET_PATH, dtype=torch.get_default_dtype())
    # PR COMMENT this is a temporary hack until kernel is properly implemented that can
    #            deal with tensor maps with different species pairs
    # for system in systems:
    #    system.species = torch.ones(len(system.species), dtype=torch.int32)

    conf = {
        "U0": {
            "quantity": "energy",
            "read_from": DATASET_PATH,
            "file_format": ".xyz",
            "key": "U0",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets = read_targets(OmegaConf.create(conf))
    dataset = Dataset(system=systems, U0=targets["U0"])

    hypers = DEFAULT_HYPERS.copy()
    hypers["training"]["num_epochs"] = 2

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        targets={
            "U0": TargetInfo(
                quantity="energy",
                unit="eV",
            ),
        },
    )
    gap = train([dataset], [dataset], dataset_info, hypers)

    # Predict on the first five systems
    output = gap(systems[:5], {"U0": gap.capabilities.outputs["U0"]})

    expected_output = torch.tensor(
        [[-40.5891], [-56.7122], [-76.4146], [-77.3364], [-93.4905]]
    )

    assert torch.allclose(output["U0"].block().values, expected_output, rtol=0.3)

    # Tests that the model is rotationally invariant
    system = ase.io.read(DATASET_PATH)
    # PR COMMENT this is a temporary hack until kernel is properly implemented that can
    #            deal with tensor maps with different species pairs
    system.numbers = np.ones(len(system.numbers))

    original_system = copy.deepcopy(system)
    system.rotate(48, "y")

    original_output = gap(
        [rascaline.torch.systems_to_torch(original_system)],
        {"U0": gap.capabilities.outputs["U0"]},
    )
    rotated_output = gap(
        [rascaline.torch.systems_to_torch(system)],
        {"U0": gap.capabilities.outputs["U0"]},
    )

    assert torch.allclose(
        original_output["U0"].block().values,
        rotated_output["U0"].block().values,
    )


def test_ethanol_regression_train_and_invariance():
    """Perform a regression test on the model when trained for 2 epoch on a small
    dataset.  We perform also the invariance test here because one needs a trained model
    for this.
    """

    systems = read_systems(DATASET_ETHANOL_PATH)
    # PR COMMENT this is a temporary hack until kernel is properly implemented that can
    #            deal with tensor maps with different species pairs
    # for system in systems:
    #    system.species = torch.ones(len(system.species), dtype=torch.int32)

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

    targets = read_targets(OmegaConf.create(conf))
    dataset = Dataset(system=systems, energy=targets["energy"])

    hypers = DEFAULT_HYPERS.copy()
    hypers["model"]["sparse_points"]["points"] = 900

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        targets={
            "U0": TargetInfo(
                quantity="energy",
                unit="eV",
            ),
        },
    )
    gap = train([dataset], [dataset], dataset_info, hypers)
    # Predict on the first five systems
    output = gap(systems[:5], {"energy": gap.capabilities.outputs["energy"]})
    # taken from the file ethanol_reduced_100.xyz
    data = ase.io.read(DATASET_ETHANOL_PATH, ":5", format="extxyz")
    expected_output = torch.tensor([[i.info["energy"]] for i in data])
    # np.savetxt(
    #    "/Users/davidetisi/Documents/Work/Software/metatensor-models/expected_forces.dat",
    #    -output["energy"].block().gradient("positions").values.reshape(45, 3),
    # )
    expected_forces = torch.vstack([torch.Tensor(i.arrays["forces"]) for i in data])
    # expected_forces = np.loadtxt(
    #    str(Path(__file__).parent.resolve() / "expected_forces.dat")
    # )
    assert torch.allclose(output["energy"].block().values, expected_output, rtol=0.1)
    assert torch.allclose(
        -output["energy"].block().gradient("positions").values.reshape(-1),
        torch.Tensor(expected_forces.reshape(-1)),
        rtol=20,
    )
    # breakpoint()
    # Tests that the model is rotationally invariant
    system = ase.io.read(DATASET_ETHANOL_PATH)
    # PR COMMENT this is a temporary hack until kernel is properly implemented that can
    #            deal with tensor maps with different species pairs
    # system.numbers = np.ones(len(system.numbers))

    original_system = copy.deepcopy(system)
    system.rotate(48, "y")

    original_output = gap(
        [rascaline.torch.systems_to_torch(original_system)],
        {"energy": gap.capabilities.outputs["energy"]},
    )
    rotated_output = gap(
        [rascaline.torch.systems_to_torch(system)],
        {"energy": gap.capabilities.outputs["energy"]},
    )

    assert torch.allclose(
        original_output["energy"].block().values,
        rotated_output["energy"].block().values,
    )
