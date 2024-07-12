import json
from pathlib import Path

import pytest
import torch
from jsonschema.exceptions import ValidationError
from metatensor.torch import Labels
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    System,
)
from pet.hypers import Hypers
from pet.pet import PET

from metatrain.experimental.pet import PET as WrappedPET
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo, TargetInfo, TargetInfoDict
from metatrain.utils.jsonschema import validate
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


DEFAULT_HYPERS = get_default_hypers("experimental.pet")
with open(Path(__file__).parents[1] / "schema-hypers.json", "r") as f:
    SCHEMA_HYPERS = json.load(f)


@pytest.mark.parametrize(
    "new_option",
    [
        "ATOMIC_BATCH_SIZE",
        "EPOCH_NUM_ATOMIC",
        "SCHEDULER_STEP_SIZE_ATOMIC",
        "EPOCHS_WARMUP_ATOMIC",
    ],
)
def test_exclusive_hypers(new_option):
    """Test that the `_ATOMIC` is mutually exclusive."""

    options = {
        "training": {
            "SCHEDULER_STEP_SIZE": 1,
            "EPOCH_NUM": 1,
            "STRUCTURAL_BATCH_SIZE": 1,
            "EPOCHS_WARMUP": 1,
        }
    }

    validate(instance=options, schema=SCHEMA_HYPERS)

    options["training"][new_option] = 1
    with pytest.raises(ValidationError, match="should not be valid under"):
        validate(instance=options, schema=SCHEMA_HYPERS)


def test_prediction():
    """Tests that the model runs without errors."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types={1, 6, 7, 8},
        targets=TargetInfoDict(energy=TargetInfo(quantity="energy", unit="eV")),
    )
    model = WrappedPET(DEFAULT_HYPERS["model"], dataset_info)
    ARCHITECTURAL_HYPERS = Hypers(model.hypers)
    raw_pet = PET(ARCHITECTURAL_HYPERS, 0.0, len(model.atomic_types))
    model.set_trained_model(raw_pet)

    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell=torch.zeros(3, 3),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

    evaluation_options = ModelEvaluationOptions(
        length_unit=dataset_info.length_unit,
        outputs={"energy": ModelOutput()},
    )

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        atomic_types=model.atomic_types,
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
        interaction_range=DEFAULT_HYPERS["model"]["N_GNN_LAYERS"]
        * DEFAULT_HYPERS["model"]["R_CUT"],
        dtype="float32",
        supported_devices=["cpu", "cuda"],
    )

    model = MetatensorAtomisticModel(model.eval(), ModelMetadata(), capabilities)
    model(
        [system],
        evaluation_options,
        check_consistency=True,
    )


def test_per_atom_predictions_functionality():
    """Tests that the model can do predictions in
    per-atom mode."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types={1, 6, 7, 8},
        targets=TargetInfoDict(energy=TargetInfo(quantity="energy", unit="eV")),
    )
    model = WrappedPET(DEFAULT_HYPERS["model"], dataset_info)
    ARCHITECTURAL_HYPERS = Hypers(model.hypers)
    raw_pet = PET(ARCHITECTURAL_HYPERS, 0.0, len(model.atomic_types))
    model.set_trained_model(raw_pet)

    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell=torch.zeros(3, 3),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

    evaluation_options = ModelEvaluationOptions(
        length_unit=dataset_info.length_unit,
        outputs={"energy": ModelOutput()},
    )

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        atomic_types=model.atomic_types,
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
                per_atom=True,
            )
        },
        interaction_range=DEFAULT_HYPERS["model"]["N_GNN_LAYERS"]
        * DEFAULT_HYPERS["model"]["R_CUT"],
        dtype="float32",
        supported_devices=["cpu", "cuda"],
    )

    model = MetatensorAtomisticModel(model.eval(), ModelMetadata(), capabilities)
    model(
        [system],
        evaluation_options,
        check_consistency=True,
    )


def test_selected_atoms_functionality():
    """Tests that the model can do predictions for a selected
    subset of the atoms in the system."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types={1, 6, 7, 8},
        targets=TargetInfoDict(energy=TargetInfo(quantity="energy", unit="eV")),
    )
    model = WrappedPET(DEFAULT_HYPERS["model"], dataset_info)
    ARCHITECTURAL_HYPERS = Hypers(model.hypers)
    raw_pet = PET(ARCHITECTURAL_HYPERS, 0.0, len(model.atomic_types))
    model.set_trained_model(raw_pet)

    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell=torch.zeros(3, 3),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

    evaluation_options = ModelEvaluationOptions(
        length_unit=dataset_info.length_unit,
        outputs={"energy": ModelOutput()},
    )

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        atomic_types=model.atomic_types,
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
        interaction_range=DEFAULT_HYPERS["model"]["N_GNN_LAYERS"]
        * DEFAULT_HYPERS["model"]["R_CUT"],
        dtype="float32",
        supported_devices=["cpu", "cuda"],
    )

    selected_atoms = Labels(
        ["system", "atom"],
        torch.tensor([[0, a] for a in range(len(system)) if a % 2 == 0]),
    )

    evaluation_options = ModelEvaluationOptions(
        length_unit=capabilities.length_unit,
        outputs=capabilities.outputs,
        selected_atoms=selected_atoms,
    )

    model = MetatensorAtomisticModel(model.eval(), ModelMetadata(), capabilities)
    model(
        [system],
        evaluation_options,
        check_consistency=True,
    )
