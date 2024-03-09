import ase
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


def test_prediction_subset():
    """Tests that the model can predict on a subset
    of the elements it was trained on."""

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
    system = ase.Atoms("O2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    system = systems_to_torch(system).to(torch.get_default_dtype())
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
    model(
        [system],
        evaluation_options,
        check_consistency=True,
    )
