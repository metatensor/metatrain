import os

import torch
import yaml

from metatensor_models.soap_bpnn import SoapBPNN


path = os.path.dirname(__file__)
hypers_path = os.path.join(path, "../default.yml")
dataset_path = os.path.join(path, "data/qm9_reduced_100.xyz")


def test_torchscript():
    """Tests that the model can be jitted."""

    all_species = [1, 6, 7, 8]
    hypers = yaml.safe_load(open(hypers_path, "r"))
    soap_bpnn = SoapBPNN(all_species, hypers).to(torch.float64)
    torch.jit.script(soap_bpnn)
