import random

import ase.io
import numpy as np
import rascaline.torch
import torch
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput
from omegaconf import OmegaConf

from metatensor.models.soap_bpnn import DEFAULT_HYPERS, Model, train
from metatensor.models.utils.data import Dataset, get_all_species
from metatensor.models.utils.data.readers import read_structures, read_targets

from . import DATASET_PATH


# reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def test_regression_init():
    """Perform a regression test on the model at initialization"""

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        species=[1, 6, 7, 8],
        outputs={
            "U0": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
    )
    soap_bpnn = Model(capabilities, DEFAULT_HYPERS["model"]).to(torch.float64)

    # Predict on the first fivestructures
    structures = ase.io.read(DATASET_PATH, ":5")

    output = soap_bpnn(
        [rascaline.torch.systems_to_torch(structure) for structure in structures],
        ["U0"],
    )
    expected_output = torch.tensor(
        [[-0.4615], [-0.4367], [-0.3004], [-0.2606], [-0.2380]],
        dtype=torch.float64,
    )

    assert torch.allclose(output["U0"].block().values, expected_output, rtol=1e-3)


def test_regression_train():
    """Perform a regression test on the model when
    trained for 2 epoch on a small dataset"""

    structures = read_structures(DATASET_PATH)

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

    dataset = Dataset(structures, targets)

    hypers = DEFAULT_HYPERS.copy()
    hypers["training"]["num_epochs"] = 2

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        species=get_all_species(dataset),
        outputs={
            "U0": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
    )
    soap_bpnn = train([dataset], [dataset], capabilities, hypers)

    # Predict on the first five structures
    output = soap_bpnn(structures[:5], ["U0"])

    expected_output = torch.tensor(
        [[-40.1358], [-56.1721], [-76.1576], [-77.1174], [-93.1679]],
        dtype=torch.float64,
    )

    assert torch.allclose(output["U0"].block().values, expected_output, rtol=1e-3)
