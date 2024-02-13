import copy

import ase.io
import rascaline.torch
import torch
from metatensor.torch.atomistic import (
    ModelCapabilities,
    ModelOutput,
    NeighborsListOptions,
)

from metatensor.models.alchemical_model import DEFAULT_HYPERS, Model

from ..utils import get_rascaline_neighbors_list
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
    original_structure = copy.deepcopy(structure)
    structure.rotate(48, "y")

    nl_options = NeighborsListOptions(model_cutoff=5.0, full_list=True)
    original_system = rascaline.torch.systems_to_torch(original_structure)
    nl = get_rascaline_neighbors_list(original_system, nl_options)
    original_system.add_neighbors_list(nl_options, nl)

    system = rascaline.torch.systems_to_torch(structure)
    nl = get_rascaline_neighbors_list(system, nl_options)
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
