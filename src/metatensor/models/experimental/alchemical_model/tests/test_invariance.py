import copy

import ase.io
import torch
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    systems_to_torch,
)

from metatensor.models.experimental.alchemical_model import DEFAULT_HYPERS, Model
from metatensor.models.utils.neighbors_lists import get_system_with_neighbors_lists

from . import DATASET_PATH


def test_rotational_invariance():
    """Tests that the model is rotationally invariant."""

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
        supported_devices=["cpu"],
    )
    alchemical_model = Model(capabilities, DEFAULT_HYPERS["model"])
    system = ase.io.read(DATASET_PATH)
    original_system = copy.deepcopy(system)
    system.rotate(48, "y")
    original_system = systems_to_torch(original_system, dtype=torch.get_default_dtype())
    original_system = get_system_with_neighbors_lists(
        original_system, alchemical_model.requested_neighbors_lists()
    )
    system = systems_to_torch(system, dtype=torch.get_default_dtype())
    system = get_system_with_neighbors_lists(
        system, alchemical_model.requested_neighbors_lists()
    )

    evaluation_options = ModelEvaluationOptions(
        length_unit=capabilities.length_unit,
        outputs=capabilities.outputs,
    )

    model = MetatensorAtomisticModel(
        alchemical_model.eval(), ModelMetadata(), alchemical_model.capabilities
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
