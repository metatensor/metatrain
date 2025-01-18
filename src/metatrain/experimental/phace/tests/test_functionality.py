import metatensor.torch
import pytest
import torch
from jsonschema.exceptions import ValidationError
from metatensor.torch.atomistic import ModelOutput, System
from omegaconf import OmegaConf

from metatrain.experimental.phace import PhACE
from metatrain.utils.architectures import check_architecture_options
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DEFAULT_HYPERS, MODEL_HYPERS


def test_prediction_subset_elements():
    """Tests that the model can predict on a subset of the elements it was trained
    on."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )

    model = torch.jit.script(PhACE(MODEL_HYPERS, dataset_info))

    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    model(
        [system],
        {"energy": model.outputs["energy"]},
    )


def test_prediction_subset_atoms():
    """Tests that the model can predict on a subset
    of the atoms in a system."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )

    model = torch.jit.script(PhACE(MODEL_HYPERS, dataset_info))

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
    get_system_with_neighbor_lists(system_monomer, model.requested_neighbor_lists())

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
    get_system_with_neighbor_lists(
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
        energy_monomer["energy"],
        energy_monomer_in_dimer["energy"],
        atol=1e-5,
        rtol=1e-5,
    )


def test_output_last_layer_features():
    """Tests that the model can output its last layer features."""
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )

    model = torch.jit.script(PhACE(MODEL_HYPERS, dataset_info))

    system = System(
        types=torch.tensor([6, 1, 8, 7]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

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
        32,
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
        1,
        32,
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
        32,
    )
    assert features.properties.names == [
        "properties",
    ]

    assert outputs["mtt::aux::energy_last_layer_features"].block().samples.names == [
        "system"
    ]
    assert outputs["mtt::aux::energy_last_layer_features"].block().values.shape == (
        1,
        1,
        32,
    )
    assert outputs["mtt::aux::energy_last_layer_features"].block().properties.names == [
        "properties",
    ]


def test_output_per_atom():
    """Tests that the model can output per-atom quantities."""
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )

    model = torch.jit.script(PhACE(MODEL_HYPERS, dataset_info))

    system = System(
        types=torch.tensor([6, 1, 8, 7]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
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
        name="experimental.phace", options=OmegaConf.to_container(hypers)
    )


def test_fixed_composition_weights_error():
    """Test that only inputd of type Dict[str, Dict[int, float]] are allowed."""
    hypers = DEFAULT_HYPERS.copy()
    hypers["training"]["fixed_composition_weights"] = {"energy": {"H": 300.0}}
    hypers = OmegaConf.create(hypers)
    with pytest.raises(ValidationError, match=r"'H' does not match '\^\[0-9\]\+\$'"):
        check_architecture_options(
            name="experimental.phace", options=OmegaConf.to_container(hypers)
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

    model = torch.jit.script(PhACE(MODEL_HYPERS, dataset_info))
    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    model(
        [system],
        {"force": model.outputs["forces"]},
    )


@pytest.mark.parametrize("per_atom", [True, False])
def test_spherical_outputs(per_atom):
    """Tests that the model can predict a spherical target with multiple blocks."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "spherical_target": get_generic_target_info(
                {
                    "quantity": "",
                    "unit": "",
                    "type": {
                        "spherical": {
                            "irreps": [
                                {"o3_lambda": 0, "o3_sigma": 1},
                                {"o3_lambda": 2, "o3_sigma": 1},
                            ]
                        }
                    },
                    "num_subtargets": 100,
                    "per_atom": per_atom,
                }
            )
        },
    )

    model = torch.jit.script(PhACE(MODEL_HYPERS, dataset_info))

    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    outputs = model(
        [system],
        {"spherical_target": model.outputs["spherical_target"]},
    )
    assert len(outputs["spherical_target"]) == 2
