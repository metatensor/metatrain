import ase.io
import rascaline.torch
import torch

from metatensor.models.soap_bpnn import (
    DEFAULT_MODEL_HYPERS,
    DEFAULT_TRAINING_HYPERS,
    Model,
    train,
)
from metatensor.models.utils.data import Dataset
from metatensor.models.utils.data.readers import read_structures, read_targets

from . import DATASET_PATH


torch.manual_seed(0)


def test_regression_init():
    """Perform a regression test on the model at initialization"""

    all_species = [1, 6, 7, 8]
    soap_bpnn = Model(all_species, DEFAULT_MODEL_HYPERS).to(torch.float64)

    # Predict on the first fivestructures
    structures = ase.io.read(DATASET_PATH, ":5")

    output = soap_bpnn(
        [rascaline.torch.systems_to_torch(structure) for structure in structures]
    )
    expected_output = torch.tensor(
        [
            [[0.5021], [0.3809], [0.1849], [0.2126], [0.0920]],
        ],
        dtype=torch.float64,
    )

    assert torch.allclose(output["energy"].block().values, expected_output, rtol=1e-3)


def test_regression_train():
    """Perform a regression test on the model when
    trained for 2 epoch on a small dataset"""

    structures = read_structures(DATASET_PATH)
    targets = read_targets(DATASET_PATH, "U0")

    dataset = Dataset(structures, targets)
    soap_bpnn = Model(dataset.all_species, DEFAULT_MODEL_HYPERS).to(torch.float64)

    hypers = DEFAULT_TRAINING_HYPERS.copy()
    hypers["num_epochs"] = 2
    train(soap_bpnn, dataset, hypers)

    # Predict on the first five structures
    output = soap_bpnn(structures[:5])

    expected_output = torch.tensor(
        [[-40.5923], [-56.5135], [-76.4457], [-77.2500], [-93.1583]],
        dtype=torch.float64,
    )

    assert torch.allclose(output["energy"].block().values, expected_output, rtol=1e-3)
