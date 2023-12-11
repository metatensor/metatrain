import copy

import ase.io
import rascaline.torch
import torch

from metatensor.models.soap_bpnn import DEFAULT_MODEL_HYPERS, Model

from . import DATASET_PATH


def test_rotational_invariance():
    """Tests that the model is rotationally invariant."""

    all_species = [1, 6, 7, 8]
    soap_bpnn = Model(all_species, DEFAULT_MODEL_HYPERS).to(torch.float64)

    structure = ase.io.read(DATASET_PATH)
    original_structure = copy.deepcopy(structure)
    structure.rotate(48, "y")

    original_output = soap_bpnn([rascaline.torch.systems_to_torch(original_structure)])
    rotated_output = soap_bpnn([rascaline.torch.systems_to_torch(structure)])

    assert torch.allclose(
        original_output["energy"].block().values,
        rotated_output["energy"].block().values,
    )
