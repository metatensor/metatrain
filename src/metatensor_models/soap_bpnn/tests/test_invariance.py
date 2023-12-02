import copy
import os

import ase.io
import rascaline.torch
import torch
import yaml

from metatensor_models.soap_bpnn import SoapBPNN


path = os.path.dirname(__file__)
hypers_path = os.path.join(path, "../default.yml")
dataset_path = os.path.join(path, "data/qm9_reduced_100.xyz")


def test_rotational_invariance():
    """Tests that the model is rotationally invariant."""

    all_species = [1, 6, 7, 8]
    hypers = yaml.safe_load(open(hypers_path, "r"))
    soap_bpnn = SoapBPNN(all_species, hypers).to(torch.float64)

    structure = ase.io.read(dataset_path)
    original_structure = copy.deepcopy(structure)
    structure.rotate(48, "y")

    original_output = soap_bpnn([rascaline.torch.systems_to_torch(original_structure)])
    rotated_output = soap_bpnn([rascaline.torch.systems_to_torch(structure)])

    assert torch.allclose(
        original_output["energy"].block().values,
        rotated_output["energy"].block().values,
    )
