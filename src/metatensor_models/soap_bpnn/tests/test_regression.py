import ase.io
import rascaline.torch
import torch
import yaml
from metatensor_models.utils.data import Dataset, collate_fn
from metatensor_models.utils.data.readers import read_structures, read_targets

from metatensor_models.soap_bpnn import SoapBPNN, train


torch.random.manual_seed(0)


def test_regression_init():
    """Perform a regression test on the model at initialization"""

    all_species = [1, 6, 7, 8, 9]
    hypers = yaml.safe_load(open("../default.yml", "r"))
    soap_bpnn = SoapBPNN(all_species, hypers).to(torch.float64)

    structures = ase.io.read("data/qm9_reduced_100.xyz", ":5")

    output = soap_bpnn(
        [rascaline.torch.systems_to_torch(structure) for structure in structures]
    )
    expected_output = torch.tensor(
        [[ 0.051100484235],
        [ 0.226915388550],
        [-0.069549073530],
        [-0.218989772242],
        [-0.042997152257]],
        dtype=torch.float64,
    )

    assert torch.allclose(output["energy"].block().values, expected_output)


def test_regression_train():
    """Perform a regression test on the model when
    trained for 2 epoch on a small dataset"""

    all_species = [1, 6, 7, 8, 9]
    hypers = yaml.safe_load(open("../default.yml", "r"))
    hypers["epochs"] = 2
    hypers["batch_size"] = 5
    soap_bpnn = SoapBPNN(all_species, hypers).to(torch.float64)

    structures = read_structures("data/qm9_reduced_100.xyz")
    targets = read_targets("data/qm9_reduced_100.xyz", "U0")

    dataset = Dataset(structures, targets)

    hypers_training = hypers["training"]
    hypers_training["num_epochs"] = 2
    train(soap_bpnn, dataset, hypers_training)

    output = soap_bpnn(structures[:5])
    expected_output = torch.tensor(
        [[-1.182792209483],
        [-0.836589440867],
        [-0.740011448717],
        [-0.896406914741],
        [-0.666903846884]],
        dtype=torch.float64,
    )

    assert torch.allclose(output["energy"].block().values, expected_output)
    

    

