import copy

import ase.io
import rascaline.torch
import torch
import yaml

from metatensor_models.soap_bpnn import SoapBPNN


def test_rotational_invariance():
    """Tests that the model is rotationally invariant."""

    all_species = [1, 6, 7, 8, 9]
    hypers = yaml.safe_load(open("../default.yml", "r"))
    soap_bpnn = SoapBPNN(all_species, hypers).to(torch.float64)

    structure = ase.io.read("data/qm9_reduced_100.xyz")
    original_structure = copy.deepcopy(structure)
    structure.rotate(48, "y")

    original_output = soap_bpnn([rascaline.torch.systems_to_torch(original_structure)])
    rotated_output = soap_bpnn([rascaline.torch.systems_to_torch(structure)])

    assert torch.allclose(
        original_output["energy"].block().values,
        rotated_output["energy"].block().values,
    )
