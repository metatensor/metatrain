import ase
import rascaline.torch
import torch
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
)

from metatensor.models.experimental.pet import DEFAULT_HYPERS, Model
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
    )

    model = Model(capabilities, DEFAULT_HYPERS["ARCHITECTURAL_HYPERS"]).to(
        torch.float64
    )
    structure = ase.Atoms("O2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    system = rascaline.torch.systems_to_torch(structure)
    system = get_system_with_neighbors_lists(system, model.requested_neighbors_lists())

    evaluation_options = ModelEvaluationOptions(
        length_unit=capabilities.length_unit,
        outputs=capabilities.outputs,
    )

    model = MetatensorAtomisticModel(model.eval(), ModelMetadata(), model.capabilities)
    model(
        [system],
        evaluation_options,
        check_consistency=True,
    )
