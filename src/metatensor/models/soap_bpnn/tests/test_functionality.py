import ase
import rascaline.torch
import torch

from metatensor.models.soap_bpnn import Model

from . import DEAFAULT_HYPERS


def test_prediction_subset():
    """Tests that the model can predict on a subset
    of the elements it was trained on."""

    all_species = [1, 6, 7, 8]
    soap_bpnn = Model(all_species, DEAFAULT_HYPERS["model"]).to(torch.float64)

    structure = ase.Atoms("O2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    soap_bpnn([rascaline.torch.systems_to_torch(structure)])
