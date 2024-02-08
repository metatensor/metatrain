import copy

import ase.io
import rascaline.torch
import torch
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput

from metatensor.models.alchemical_model import DEFAULT_HYPERS, Model

from ..utils import get_primitive_neighbors_list
from . import DATASET_PATH


def test_rotational_invariance():
    """Tests that the model is rotationally invariant."""

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        species=[1, 6, 7, 8],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
    )
    alchemical_model = Model(capabilities, DEFAULT_HYPERS["model"]).to(torch.float64)

    structure = ase.io.read(DATASET_PATH)
    nl, nl_options = get_primitive_neighbors_list(structure)

    original_structure = copy.deepcopy(structure)
    structure.rotate(48, "y")

    original_system = rascaline.torch.systems_to_torch(original_structure)
    original_system.add_neighbors_list(nl_options, nl)
    system = rascaline.torch.systems_to_torch(structure)
    system.add_neighbors_list(nl_options, nl)

    original_output = alchemical_model(
        [original_system],
        {"energy": alchemical_model.capabilities.outputs["energy"]},
    )
    rotated_output = alchemical_model(
        [system],
        {"energy": alchemical_model.capabilities.outputs["energy"]},
    )

    assert torch.allclose(
        original_output["energy"].block().values,
        rotated_output["energy"].block().values,
    )
