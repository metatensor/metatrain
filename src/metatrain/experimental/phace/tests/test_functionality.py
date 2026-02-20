import copy

import pytest
import torch
from metatomic.torch import ModelOutput, System

from metatrain.experimental.phace import PhACE
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import MODEL_HYPERS


def _make_hypers() -> dict:
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["num_element_channels"] = 2
    hypers["num_gnn_layers"] = 1
    hypers["num_tensor_products"] = 2
    # max_eigenvalue=25.0 gives l_max=2, needed for rank-2 Cartesian targets
    hypers["radial_basis"]["max_eigenvalue"] = 25.0
    hypers["radial_basis"]["mlp_expansion_ratio"] = 1
    hypers["radial_basis"]["mlp_depth"] = 2
    hypers["mlp_head_expansion_ratio"] = 1
    return hypers


def _make_system(model: PhACE) -> System:
    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        cell=2.0 * torch.eye(3),
        pbc=torch.tensor([True, True, True]),
    )
    return get_system_with_neighbor_lists(system, model.requested_neighbor_lists())


def test_cartesian_rank1():
    """Tests rank-1 Cartesian (e.g. dipole) prediction."""
    hypers = _make_hypers()
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
    model = PhACE(hypers, dataset_info)
    system = _make_system(model)
    output = model([system], {"dipole": ModelOutput(per_atom=False)})
    values = output["dipole"].block().values
    # shape: (n_structures, 3, n_properties)
    assert values.shape == (1, 3, 1)


@pytest.mark.parametrize("per_atom", [True, False])
def test_nc_stress(per_atom):
    """Tests that rank-2 Cartesian non-conservative stress is symmetric."""
    hypers = _make_hypers()
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
    model = PhACE(hypers, dataset_info)
    system = _make_system(model)
    outputs = {"non_conservative_stress": ModelOutput(per_atom=per_atom)}
    stress = model([system], outputs)["non_conservative_stress"].block().values
    assert torch.allclose(stress, stress.transpose(-3, -2))


def test_multiple_targets():
    """Tests simultaneous prediction of energy + rank-1 dipole + rank-2 stress."""
    hypers = _make_hypers()
    targets = {
        "energy": get_energy_target_info("energy", {"unit": "eV"}),
        "dipole": get_generic_target_info(
            "dipole",
            {
                "quantity": "dipole",
                "unit": "",
                "type": {"cartesian": {"rank": 1}},
                "num_subtargets": 1,
                "per_atom": False,
            },
        ),
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
    model = PhACE(hypers, dataset_info)
    system = _make_system(model)
    outputs = {
        "energy": ModelOutput(quantity="energy", unit="eV", per_atom=False),
        "dipole": ModelOutput(per_atom=False),
        "non_conservative_stress": ModelOutput(per_atom=False),
    }
    result = model([system], outputs)
    assert "energy" in result
    assert "dipole" in result
    assert "non_conservative_stress" in result
    assert result["energy"].block().values.shape == (1, 1)
    assert result["dipole"].block().values.shape == (1, 3, 1)
    stress = result["non_conservative_stress"].block().values
    assert stress.shape == (1, 3, 3, 1)
    assert torch.allclose(stress, stress.transpose(-3, -2))
