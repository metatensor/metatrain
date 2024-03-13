import random

import ase.io
import numpy as np
import torch
from metatensor.learn.data import Dataset
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput, systems_to_torch
from omegaconf import OmegaConf

from metatensor.models.experimental.soap_bpnn import DEFAULT_HYPERS, Model, train
from metatensor.models.utils.data import DatasetInfo, TargetInfo
from metatensor.models.utils.data.readers import read_systems, read_targets

from . import DATASET_PATH


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
    soap_bpnn = Model(capabilities, DEFAULT_HYPERS["model"])

    # Predict on the first five systems
    systems = ase.io.read(DATASET_PATH, ":5")

    output = soap_bpnn(
        [
            systems_to_torch(system, dtype=torch.get_default_dtype())
            for system in systems
        ],
        {"U0": soap_bpnn.capabilities.outputs["U0"]},
    )
    expected_output = torch.tensor([[0.0739], [0.0758], [0.1782], [-0.3517], [-0.3251]])

    assert torch.allclose(output["U0"].block().values, expected_output, rtol=1e-3)


def test_regression_train():
    """Perform a regression test on the model when
    trained for 2 epoch on a small dataset"""

    systems = read_systems(DATASET_PATH, dtype=torch.get_default_dtype())

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
    targets = read_targets(OmegaConf.create(conf), dtype=torch.get_default_dtype())
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
    soap_bpnn = train([dataset], [dataset], dataset_info, hypers)

    # Predict on the first five systems
    output = soap_bpnn(systems[:5], {"U0": soap_bpnn.capabilities.outputs["U0"]})

    expected_output = torch.tensor(
        [[-40.3951], [-56.4275], [-76.4008], [-77.3751], [-93.4227]]
    )

    assert torch.allclose(output["U0"].block().values, expected_output, rtol=1e-3)
