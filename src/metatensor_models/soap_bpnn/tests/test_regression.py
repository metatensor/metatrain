import os

import ase.io
import rascaline.torch
import torch
import yaml

from metatensor_models.soap_bpnn import SoapBPNN, train
from metatensor_models.utils.data import Dataset
from metatensor_models.utils.data.readers import read_structures, read_targets


torch.manual_seed(0)

path = os.path.dirname(__file__)
hypers_path = os.path.join(path, "../default.yml")
dataset_path = os.path.join(path, "data/qm9_reduced_100.xyz")


def test_regression_init():
    """Perform a regression test on the model at initialization"""

    all_species = [1, 6, 7, 8]
    hypers = yaml.safe_load(open(hypers_path, "r"))
    soap_bpnn = SoapBPNN(all_species, hypers).to(torch.float64)

    structures = ase.io.read(dataset_path, ":5")

    output = soap_bpnn(
        [rascaline.torch.systems_to_torch(structure) for structure in structures]
    )
    expected_output = torch.tensor(
        [
            [-0.236301918363],
            [-0.168581936238],
            [-0.130610894525],
            [-0.357959424353],
            [-0.290683367111],
        ],
        dtype=torch.float64,
    )

    assert torch.allclose(output["energy"].block().values, expected_output)


def test_regression_train():
    """Perform a regression test on the model when
    trained for 2 epoch on a small dataset"""

    all_species = [1, 6, 7, 8]
    hypers = yaml.safe_load(open(hypers_path, "r"))
    hypers["epochs"] = 2
    hypers["batch_size"] = 5
    soap_bpnn = SoapBPNN(all_species, hypers).to(torch.float64)

    structures = read_structures(dataset_path)
    targets = read_targets(dataset_path, "U0")

    dataset = Dataset(structures, targets)

    hypers_training = hypers["training"]
    hypers_training["num_epochs"] = 2
    train(soap_bpnn, dataset, hypers_training)

    output = soap_bpnn(structures[:5])
    expected_output = torch.tensor(
        [
            [-40.250192670994],
            [-56.248591125262],
            [-76.192451563486],
            [-77.000614265071],
            [-92.977346257705],
        ],
        dtype=torch.float64,
    )

    assert torch.allclose(output["energy"].block().values, expected_output)
