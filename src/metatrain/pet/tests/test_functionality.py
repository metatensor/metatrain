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

from metatrain.pet import PET as WrappedPET
from metatrain.pet.modules.hypers import Hypers
from metatrain.pet.modules.pet import PET
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)
from metatrain.utils.jsonschema import validate
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)


DEFAULT_HYPERS = get_default_hypers("pet")
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
        atomic_types=[1, 6, 7, 8],
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )
    model = WrappedPET(DEFAULT_HYPERS["model"], dataset_info)
    ARCHITECTURAL_HYPERS = Hypers(model.hypers)
    raw_pet = PET(ARCHITECTURAL_HYPERS, 0.0, len(model.atomic_types))
    model.set_trained_model(raw_pet)

    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    requested_neighbor_lists = get_requested_neighbor_lists(model)
    system = get_system_with_neighbor_lists(system, requested_neighbor_lists)

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
        atomic_types=[1, 6, 7, 8],
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )
    model = WrappedPET(DEFAULT_HYPERS["model"], dataset_info)
    ARCHITECTURAL_HYPERS = Hypers(model.hypers)
    raw_pet = PET(ARCHITECTURAL_HYPERS, 0.0, len(model.atomic_types))
    model.set_trained_model(raw_pet)

    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    requested_neighbor_lists = get_requested_neighbor_lists(model)
    system = get_system_with_neighbor_lists(system, requested_neighbor_lists)

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
        atomic_types=[1, 6, 7, 8],
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )
    model = WrappedPET(DEFAULT_HYPERS["model"], dataset_info)
    ARCHITECTURAL_HYPERS = Hypers(model.hypers)
    raw_pet = PET(ARCHITECTURAL_HYPERS, 0.0, len(model.atomic_types))
    model.set_trained_model(raw_pet)

    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    requested_neighbor_lists = get_requested_neighbor_lists(model)
    system = get_system_with_neighbor_lists(system, requested_neighbor_lists)

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


@pytest.mark.parametrize("per_atom", [True, False])
def test_vector_output(per_atom):
    """Tests that the model can predict a (spherical) vector output."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "forces": get_generic_target_info(
                {
                    "quantity": "forces",
                    "unit": "",
                    "type": {
                        "spherical": {"irreps": [{"o3_lambda": 1, "o3_sigma": 1}]}
                    },
                    "num_subtargets": 100,
                    "per_atom": per_atom,
                }
            )
        },
    )

    with pytest.raises(ValueError, match="PET only supports total-energy-like outputs"):
        WrappedPET(DEFAULT_HYPERS["model"], dataset_info)


def test_output_features():
    """Tests that the model can output its features and last-layer features."""
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )

    model = WrappedPET(DEFAULT_HYPERS["model"], dataset_info)
    ARCHITECTURAL_HYPERS = Hypers(model.hypers)
    raw_pet = PET(ARCHITECTURAL_HYPERS, 0.0, len(model.atomic_types))
    model.set_trained_model(raw_pet)

    system = System(
        types=torch.tensor([6, 1, 8, 7]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )

    requested_neighbor_lists = get_requested_neighbor_lists(model)
    system = get_system_with_neighbor_lists(system, requested_neighbor_lists)

    # last-layer features per atom:
    ll_output_options = ModelOutput(
        quantity="",
        unit="unitless",
        per_atom=True,
    )
    outputs = model(
        [system],
        {
            "energy": ModelOutput(quantity="energy", unit="eV", per_atom=True),
            "mtt::aux::energy_last_layer_features": ll_output_options,
            "features": ll_output_options,
        },
    )
    assert "energy" in outputs
    assert "mtt::aux::energy_last_layer_features" in outputs
    assert "features" in outputs
    last_layer_features = outputs["mtt::aux::energy_last_layer_features"].block()
    features = outputs["features"].block()
    assert last_layer_features.samples.names == ["system", "atom"]
    assert last_layer_features.values.shape == (
        4,
        768,  # 768 = 3 (gnn layers) * 256 (128 for edge repr, 128 for node repr)
    )
    assert last_layer_features.properties.names == ["properties"]
    assert features.samples.names == ["system", "atom"]
    assert features.values.shape == (
        4,
        768,  # 768 = 3 (gnn layers) * 256 (128 for edge repr, 128 for node repr)
    )
    assert features.properties.names == ["properties"]

    # last-layer features per system:
    ll_output_options = ModelOutput(
        quantity="",
        unit="unitless",
        per_atom=False,
    )
    outputs = model(
        [system],
        {
            "energy": ModelOutput(quantity="energy", unit="eV", per_atom=True),
            "mtt::aux::energy_last_layer_features": ll_output_options,
            "features": ll_output_options,
        },
    )
    assert "energy" in outputs
    assert "mtt::aux::energy_last_layer_features" in outputs
    assert "features" in outputs
    assert outputs["mtt::aux::energy_last_layer_features"].block().samples.names == [
        "system"
    ]
    assert outputs["mtt::aux::energy_last_layer_features"].block().values.shape == (
        1,
        768,  # 768 = 3 (gnn layers) * 256 (128 for edge repr, 128 for node repr)
    )
    assert outputs["mtt::aux::energy_last_layer_features"].block().properties.names == [
        "properties",
    ]
    assert outputs["features"].block().samples.names == ["system"]
    assert outputs["features"].block().values.shape == (
        1,
        768,  # 768 = 3 (gnn layers) * 256 (128 for edge repr, 128 for node repr)
    )
    assert outputs["features"].block().properties.names == ["properties"]
