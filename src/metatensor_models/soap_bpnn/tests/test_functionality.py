import os

import ase
import rascaline.torch
import torch
import yaml

from metatensor_models.soap_bpnn import SoapBPNN


path = os.path.dirname(__file__)
hypers_path = os.path.join(path, "../default.yml")
dataset_path = os.path.join(path, "data/qm9_reduced_100.xyz")


def test_prediction_subset():
    """Tests that the model can predict on a subset
    of the elements it was trained on."""

    all_species = [1, 6, 7, 8]
    hypers = yaml.safe_load(open(hypers_path, "r"))
    soap_bpnn = SoapBPNN(all_species, hypers).to(torch.float64)

    structure = ase.Atoms("O2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    soap_bpnn([rascaline.torch.systems_to_torch(structure)])
