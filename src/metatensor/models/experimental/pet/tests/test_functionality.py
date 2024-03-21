import ase
import torch
from metatensor.torch import Labels
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    systems_to_torch,
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
        supported_devices=["cpu"],
    )

    model = Model(capabilities, DEFAULT_HYPERS["ARCHITECTURAL_HYPERS"])
    structure = ase.Atoms("O2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    system = systems_to_torch(structure)
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


def test_per_atom_predictions_functionality():
    """Tests that the model can predict on a subset
    of the elements it was trained on."""

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
                per_atom=True,
            )
        },
    )

    model = Model(capabilities, DEFAULT_HYPERS["ARCHITECTURAL_HYPERS"])
    structure = ase.Atoms("O2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    system = systems_to_torch(structure)
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


def test_selected_atoms_functionality():
    """Tests that the model can predict on a subset
    of the elements it was trained on."""

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
                per_atom=True,
            )
        },
    )

    model = Model(capabilities, DEFAULT_HYPERS["ARCHITECTURAL_HYPERS"]).to(
        torch.float64
    )
    structure = ase.Atoms(
        "O3", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]
    )
    system = systems_to_torch(structure)
    system = get_system_with_neighbors_lists(system, model.requested_neighbors_lists())

    selected_atoms = Labels(
        ["system", "atom"],
        torch.tensor([[0, a] for a in range(len(system)) if a % 2 == 0]),
    )

    evaluation_options = ModelEvaluationOptions(
        length_unit=capabilities.length_unit,
        outputs=capabilities.outputs,
        selected_atoms=selected_atoms,
    )

    model = MetatensorAtomisticModel(model.eval(), ModelMetadata(), model.capabilities)
    model(
        [system],
        evaluation_options,
        check_consistency=True,
    )
