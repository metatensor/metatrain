import copy

import pytest
import torch
from metatomic.torch import ModelOutput, System

from metatrain.soap_bpnn import SoapBpnn
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import MODEL_HYPERS


def _make_hypers(legacy: bool = True, add_lambda_basis: bool = True) -> dict:
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["legacy"] = legacy
    hypers["add_lambda_basis"] = add_lambda_basis
    hypers["soap"]["max_angular"] = 2
    hypers["soap"]["max_radial"] = 2
    hypers["bpnn"]["num_neurons_per_layer"] = 4
    hypers["bpnn"]["num_hidden_layers"] = 1
    return hypers


def _make_system(model: SoapBpnn) -> System:
    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        cell=2.0 * torch.eye(3),
        pbc=torch.tensor([True, True, True]),
    )
    return get_system_with_neighbor_lists(system, model.requested_neighbor_lists())


@pytest.mark.parametrize("legacy", [True, False])
def test_scalar_output(legacy):
    """Tests scalar energy prediction for both legacy and modern paths."""
    hypers = _make_hypers(legacy)
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[6],
        targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
    )
    model = SoapBpnn(hypers, dataset_info)
    system = _make_system(model)
    output = model(
        [system], {"energy": ModelOutput(quantity="energy", unit="eV", per_atom=False)}
    )
    values = output["energy"].block().values
    assert values.shape == (1, 1)


@pytest.mark.parametrize("legacy", [True, False])
def test_cartesian_rank1(legacy):
    """Tests rank-1 Cartesian (e.g. dipole) prediction."""
    hypers = _make_hypers(legacy)
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[6],
        targets={
            "dipole": get_generic_target_info(
                "dipole",
                {
                    "quantity": "dipole",
                    "unit": "",
                    "type": {"cartesian": {"rank": 1}},
                    "num_subtargets": 1,
                    "per_atom": False,
                },
            )
        },
    )
    model = SoapBpnn(hypers, dataset_info)
    system = _make_system(model)
    output = model([system], {"dipole": ModelOutput(per_atom=False)})
    values = output["dipole"].block().values
    # shape: (n_structures, 3, n_properties)
    assert values.shape == (1, 3, 1)


@pytest.mark.parametrize("add_lambda_basis", [True, False])
@pytest.mark.parametrize("legacy", [True, False])
@pytest.mark.parametrize("per_atom", [True, False])
def test_nc_stress(legacy, per_atom, add_lambda_basis):
    """Tests that rank-2 Cartesian NC stress is symmetric."""
    hypers = _make_hypers(legacy, add_lambda_basis)
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[6],
        targets={
            "non_conservative_stress": get_generic_target_info(
                "non_conservative_stress",
                {
                    "quantity": "stress",
                    "unit": "",
                    "type": {"cartesian": {"rank": 2}},
                    "num_subtargets": 1,
                    "per_atom": per_atom,
                },
            )
        },
    )
    model = SoapBpnn(hypers, dataset_info)
    system = _make_system(model)
    outputs = {"non_conservative_stress": ModelOutput(per_atom=per_atom)}
    stress = model([system], outputs)["non_conservative_stress"].block().values
    assert torch.allclose(stress, stress.transpose(-3, -2))


@pytest.mark.parametrize("add_lambda_basis", [True, False])
@pytest.mark.parametrize("legacy", [True, False])
def test_spherical_output(legacy, add_lambda_basis):
    """Tests spherical tensor (l=1, sigma=1) prediction."""
    hypers = _make_hypers(legacy, add_lambda_basis)
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[6],
        targets={
            "spherical_target": get_generic_target_info(
                "spherical_target",
                {
                    "quantity": "",
                    "unit": "",
                    "type": {
                        "spherical": {
                            "irreps": [{"o3_lambda": 1, "o3_sigma": 1}],
                        }
                    },
                    "num_subtargets": 1,
                    "per_atom": False,
                },
            )
        },
    )
    model = SoapBpnn(hypers, dataset_info)
    system = _make_system(model)
    output = model([system], {"spherical_target": ModelOutput(per_atom=False)})
    tmap = output["spherical_target"]
    assert len(tmap.keys) == 1
    assert tmap.keys.names == ["o3_lambda", "o3_sigma"]
    assert tmap.keys.values[0].tolist() == [1, 1]
    assert tmap.block(0).values.shape == (1, 3, 1)


@pytest.mark.parametrize("add_lambda_basis", [True, False])
def test_mlp_head(add_lambda_basis):
    """
    Tests scalar prediction with an MLP head (legacy only, MLPHeadMap is per-species).
    """
    hypers = _make_hypers(legacy=True, add_lambda_basis=add_lambda_basis)
    hypers["heads"] = {"energy": "mlp"}
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[6],
        targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
    )
    model = SoapBpnn(hypers, dataset_info)
    system = _make_system(model)
    output = model(
        [system], {"energy": ModelOutput(quantity="energy", unit="eV", per_atom=False)}
    )
    values = output["energy"].block().values
    assert values.shape == (1, 1)


@pytest.mark.parametrize("add_lambda_basis", [True, False])
@pytest.mark.parametrize("legacy", [True, False])
def test_multiple_targets(legacy, add_lambda_basis):
    """Tests simultaneous prediction of energy + rank-2 Cartesian stress."""
    hypers = _make_hypers(legacy, add_lambda_basis)
    targets = {
        "energy": get_energy_target_info("energy", {"unit": "eV"}),
        "non_conservative_stress": get_generic_target_info(
            "non_conservative_stress",
            {
                "quantity": "stress",
                "unit": "",
                "type": {"cartesian": {"rank": 2}},
                "num_subtargets": 1,
                "per_atom": False,
            },
        ),
    }
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[6],
        targets=targets,
    )
    model = SoapBpnn(hypers, dataset_info)
    system = _make_system(model)
    outputs = {
        "energy": ModelOutput(quantity="energy", unit="eV", per_atom=False),
        "non_conservative_stress": ModelOutput(per_atom=False),
    }
    result = model([system], outputs)
    assert "energy" in result
    assert "non_conservative_stress" in result
    assert result["energy"].block().values.shape == (1, 1)
    stress = result["non_conservative_stress"].block().values
    assert stress.shape == (1, 3, 3, 1)
    assert torch.allclose(stress, stress.transpose(-3, -2))
