import metatensor.torch
import pytest
import torch
from jsonschema.exceptions import ValidationError
from metatensor.torch.atomistic import ModelOutput, System
from omegaconf import OmegaConf

from metatrain.experimental.nativepet import NativePET
from metatrain.experimental.nativepet.modules.transformer import AttentionBlock
from metatrain.utils.architectures import check_architecture_options
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DEFAULT_HYPERS, MODEL_HYPERS


def test_prediction():
    """Tests the basic functionality of the forward pass of the model."""
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info({"quantity": "energy", "unit": "eV"})
        },
    )

    model = NativePET(MODEL_HYPERS, dataset_info)

    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    outputs = {"energy": ModelOutput(per_atom=False)}
    model([system, system], outputs)


def test_pet_padding():
    """Tests that the model predicts the same energy independently of the
    padding size."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info({"quantity": "energy", "unit": "eV"})
        },
    )

    model = NativePET(MODEL_HYPERS, dataset_info)

    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    outputs = {"energy": ModelOutput(per_atom=False)}
    lone_output = model([system], outputs)

    system_2 = System(
        types=torch.tensor([6, 6, 6, 6, 6, 6, 6]),
        positions=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 3.0],
                [0.0, 0.0, 4.0],
                [0.0, 0.0, 5.0],
                [0.0, 0.0, 6.0],
            ]
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system_2 = get_system_with_neighbor_lists(
        system_2, model.requested_neighbor_lists()
    )
    padded_output = model([system, system_2], outputs)

    lone_energy = lone_output["energy"].block().values.squeeze(-1)[0]
    padded_energy = padded_output["energy"].block().values.squeeze(-1)[0]

    assert torch.allclose(lone_energy, padded_energy, atol=1e-6, rtol=1e-6)


def test_prediction_subset_elements():
    """Tests that the model can predict on a subset of the elements it was trained
    on."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info({"quantity": "energy", "unit": "eV"})
        },
    )

    model = NativePET(MODEL_HYPERS, dataset_info)

    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    model(
        [system],
        {"energy": model.outputs["energy"]},
    )


def test_prediction_subset_atoms():
    """Tests that the model can predict on a subset
    of the atoms in a system."""

    # we need float64 for this test, then we will change it back at the end
    default_dtype_before = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info({"quantity": "energy", "unit": "eV"})
        },
    )

    model = NativePET(MODEL_HYPERS, dataset_info)

    # Since we don't yet support atomic predictions, we will test this by
    # predicting on a system with two monomers at a large distance

    system_monomer = System(
        types=torch.tensor([7, 8, 8]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system_monomer = get_system_with_neighbor_lists(
        system_monomer, model.requested_neighbor_lists()
    )

    energy_monomer = model(
        [system_monomer],
        {"energy": ModelOutput(per_atom=False)},
    )

    system_far_away_dimer = System(
        types=torch.tensor([7, 7, 8, 8, 8, 8]),
        positions=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 50.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
                [0.0, 51.0, 0.0],
                [0.0, 42.0, 0.0],
            ],
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system_far_away_dimer = get_system_with_neighbor_lists(
        system_far_away_dimer, model.requested_neighbor_lists()
    )

    selection_labels = metatensor.torch.Labels(
        names=["system", "atom"],
        values=torch.tensor([[0, 0], [0, 2], [0, 3]]),
    )

    energy_dimer = model(
        [system_far_away_dimer],
        {"energy": ModelOutput(per_atom=False)},
    )

    energy_monomer_in_dimer = model(
        [system_far_away_dimer],
        {"energy": ModelOutput(per_atom=False)},
        selected_atoms=selection_labels,
    )

    assert not metatensor.torch.allclose(
        energy_monomer["energy"], energy_dimer["energy"]
    )

    assert metatensor.torch.allclose(
        energy_monomer["energy"], energy_monomer_in_dimer["energy"]
    )

    torch.set_default_dtype(default_dtype_before)


def test_output_last_layer_features():
    """Tests that the model can output its last layer features."""
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info({"quantity": "energy", "unit": "eV"})
        },
    )

    model = NativePET(MODEL_HYPERS, dataset_info)

    system = System(
        types=torch.tensor([6, 1, 8, 7]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

    # last-layer features per atom:
    ll_output_options = ModelOutput(
        quantity="",
        unit="unitless",
        per_atom=True,
    )
    outputs = model(
        [system],
        {
            "energy": model.outputs["energy"],
            "features": ll_output_options,
            "mtt::aux::energy_last_layer_features": ll_output_options,
        },
    )
    assert "energy" in outputs
    assert "features" in outputs
    assert "mtt::aux::energy_last_layer_features" in outputs

    features = outputs["features"].block()
    assert features.samples.names == [
        "system",
        "atom",
    ]
    assert features.values.shape == (
        4,
        MODEL_HYPERS["d_pet"] * MODEL_HYPERS["num_gnn_layers"] * 2,
    )
    assert features.properties.names == [
        "properties",
    ]

    last_layer_features = outputs["mtt::aux::energy_last_layer_features"].block()
    assert last_layer_features.samples.names == [
        "system",
        "atom",
    ]
    assert last_layer_features.values.shape == (
        4,
        MODEL_HYPERS["d_head"] * MODEL_HYPERS["num_gnn_layers"] * 2,
    )
    assert last_layer_features.properties.names == [
        "properties",
    ]

    # last-layer features per system:
    ll_output_options = ModelOutput(
        quantity="",
        unit="unitless",
        per_atom=False,
    )
    outputs = model(
        [system],
        {
            "energy": model.outputs["energy"],
            "features": ll_output_options,
            "mtt::aux::energy_last_layer_features": ll_output_options,
        },
    )
    assert "energy" in outputs
    assert "features" in outputs
    assert "mtt::aux::energy_last_layer_features" in outputs

    features = outputs["features"].block()
    assert features.samples.names == [
        "system",
    ]
    assert features.values.shape == (
        1,
        MODEL_HYPERS["d_pet"] * MODEL_HYPERS["num_gnn_layers"] * 2,
    )

    assert outputs["mtt::aux::energy_last_layer_features"].block().samples.names == [
        "system"
    ]
    assert outputs["mtt::aux::energy_last_layer_features"].block().values.shape == (
        1,
        MODEL_HYPERS["d_head"] * MODEL_HYPERS["num_gnn_layers"] * 2,
    )
    assert outputs["mtt::aux::energy_last_layer_features"].block().properties.names == [
        "properties",
    ]


def test_output_per_atom():
    """Tests that the model can output per-atom quantities."""
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info({"quantity": "energy", "unit": "eV"})
        },
    )

    model = NativePET(MODEL_HYPERS, dataset_info)

    system = System(
        types=torch.tensor([6, 1, 8, 7]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

    outputs = model(
        [system],
        {"energy": model.outputs["energy"]},
    )

    assert outputs["energy"].block().samples.names == ["system", "atom"]
    assert outputs["energy"].block().values.shape == (4, 1)


def test_fixed_composition_weights():
    """Tests the correctness of the json schema for fixed_composition_weights"""

    hypers = DEFAULT_HYPERS.copy()
    hypers["training"]["fixed_composition_weights"] = {
        "energy": {
            1: 1.0,
            6: 0.0,
            7: 0.0,
            8: 0.0,
            9: 3000.0,
        }
    }
    hypers = OmegaConf.create(hypers)
    check_architecture_options(
        name="experimental.nativepet", options=OmegaConf.to_container(hypers)
    )


def test_fixed_composition_weights_error():
    """Test that only inputd of type Dict[str, Dict[int, float]] are allowed."""
    hypers = DEFAULT_HYPERS.copy()
    hypers["training"]["fixed_composition_weights"] = {"energy": {"H": 300.0}}
    hypers = OmegaConf.create(hypers)
    with pytest.raises(ValidationError, match=r"'H' does not match '\^\[0-9\]\+\$'"):
        check_architecture_options(
            name="experimental.nativepet", options=OmegaConf.to_container(hypers)
        )


@pytest.mark.parametrize("per_atom", [True, False])
def test_vector_output(per_atom):
    """Tests that the model can predict a Cartesian vector output."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "forces": get_generic_target_info(
                {
                    "quantity": "forces",
                    "unit": "",
                    "type": {"cartesian": {"rank": 1}},
                    "num_subtargets": 100,
                    "per_atom": per_atom,
                }
            )
        },
    )

    model = NativePET(MODEL_HYPERS, dataset_info)

    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    model(
        [system],
        {"force": model.outputs["forces"]},
    )


@pytest.mark.parametrize("per_atom", [True, False])
def test_spherical_output(per_atom):
    """Tests that the model can predict a spherical tensor output."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "spherical_tensor": get_generic_target_info(
                {
                    "quantity": "spherical_tensor",
                    "unit": "",
                    "type": {
                        "spherical": {"irreps": [{"o3_lambda": 2, "o3_sigma": 1}]}
                    },
                    "num_subtargets": 100,
                    "per_atom": per_atom,
                }
            )
        },
    )

    model = NativePET(MODEL_HYPERS, dataset_info)

    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    model(
        [system],
        {"spherical_tensor": model.outputs["spherical_tensor"]},
    )


@pytest.mark.parametrize("per_atom", [True, False])
def test_spherical_output_multi_block(per_atom):
    """Tests that the model can predict a spherical tensor output
    with multiple irreps."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "spherical_tensor": get_generic_target_info(
                {
                    "quantity": "spherical_tensor",
                    "unit": "",
                    "type": {
                        "spherical": {
                            "irreps": [
                                {"o3_lambda": 2, "o3_sigma": 1},
                                {"o3_lambda": 1, "o3_sigma": 1},
                                {"o3_lambda": 0, "o3_sigma": 1},
                            ]
                        }
                    },
                    "num_subtargets": 100,
                    "per_atom": per_atom,
                }
            )
        },
    )

    model = NativePET(MODEL_HYPERS, dataset_info)

    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    outputs = model(
        [system],
        {"spherical_tensor": model.outputs["spherical_tensor"]},
    )
    assert len(outputs["spherical_tensor"]) == 3


def test_consistency():
    """Tests that the two implementations of attention are consistent."""

    num_centers = 100
    num_neighbors_per_center = 50
    hidden_size = 128
    num_heads = 4

    attention = AttentionBlock(hidden_size, num_heads)

    inputs = torch.randn(num_centers, num_neighbors_per_center, hidden_size)
    radial_mask = torch.rand(
        num_centers, num_neighbors_per_center, num_neighbors_per_center
    )

    attention_output_torch = attention(inputs, radial_mask, use_manual_attention=False)
    attention_output_manual = attention(inputs, radial_mask, use_manual_attention=True)

    assert torch.allclose(attention_output_torch, attention_output_manual, atol=1e-6)
