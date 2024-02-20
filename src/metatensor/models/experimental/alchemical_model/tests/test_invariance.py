import copy

import ase.io
import rascaline.torch
import torch
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelOutput,
)

from metatensor.models.experimental.alchemical_model import DEFAULT_HYPERS, Model
from metatensor.models.utils.neighbors_lists import get_system_with_neighbors_lists

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
    original_system = rascaline.torch.systems_to_torch(original_structure)
    original_system = get_system_with_neighbors_lists(
        original_system, alchemical_model.requested_neighbors_lists()
    )
    system = rascaline.torch.systems_to_torch(structure)
    system = get_system_with_neighbors_lists(
        system, alchemical_model.requested_neighbors_lists()
    )

    evaluation_options = ModelEvaluationOptions(
        length_unit=capabilities.length_unit,
        outputs=capabilities.outputs,
    )

    model = MetatensorAtomisticModel(
        alchemical_model.eval(), alchemical_model.capabilities
    )
    original_output = model(
        [original_system],
        evaluation_options,
        check_consistency=True,
    )
    rotated_output = model(
        [system],
        evaluation_options,
        check_consistency=True,
    )

    assert torch.allclose(
        original_output["energy"].block().values,
        rotated_output["energy"].block().values,
    )
