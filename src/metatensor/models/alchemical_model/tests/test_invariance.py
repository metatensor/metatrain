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

from metatensor.models.alchemical_model import DEFAULT_HYPERS, Model
from metatensor.models.alchemical_model.utils.neighbors_lists import (
    get_rascaline_neighbors_list,
)

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
    requested_neighbors_lists = alchemical_model.requested_neighbors_lists()

    structure = ase.io.read(DATASET_PATH)
    original_structure = copy.deepcopy(structure)
    structure.rotate(48, "y")
    original_system = rascaline.torch.systems_to_torch(original_structure)
    system = rascaline.torch.systems_to_torch(structure)
    for nl_options in requested_neighbors_lists:
        nl = get_rascaline_neighbors_list(original_system, nl_options)
        original_system.add_neighbors_list(nl_options, nl)

        nl = get_rascaline_neighbors_list(system, nl_options)
        system.add_neighbors_list(nl_options, nl)

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
