import pytest
from omegaconf import DictConfig

from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)


@pytest.fixture
def energy_target_config() -> DictConfig:
    return DictConfig(
        {
            "quantity": "energy",
            "unit": "eV",
            "per_atom": False,
            "num_properties": 1,
            "type": "scalar",
        }
    )


@pytest.fixture
def scalar_target_config() -> DictConfig:
    return DictConfig(
        {
            "quantity": "scalar",
            "unit": "",
            "per_atom": False,
            "num_properties": 10,
            "type": "scalar",
        }
    )


@pytest.fixture
def cartesian_target_config() -> DictConfig:
    return DictConfig(
        {
            "quantity": "dipole",
            "unit": "D",
            "per_atom": True,
            "num_properties": 5,
            "type": {
                "Cartesian": {
                    "rank": 1,
                }
            },
        }
    )


@pytest.fixture
def spherical_target_config() -> DictConfig:
    return DictConfig(
        {
            "quantity": "spherical",
            "unit": "",
            "per_atom": False,
            "num_properties": 1,
            "type": {
                "spherical": [
                    {"o3_lambda": 0, "o3_sigma": 1},
                    {"o3_lambda": 2, "o3_sigma": 1},
                ],
            },
        }
    )


def test_layout_energy(energy_target_config):

    target_info = get_energy_target_info(energy_target_config)
    assert target_info.quantity == "energy"
    assert target_info.unit == "eV"
    assert target_info.per_atom is False
    assert target_info.gradients == []

    target_info = get_energy_target_info(
        energy_target_config, add_position_gradients=True
    )
    assert target_info.quantity == "energy"
    assert target_info.unit == "eV"
    assert target_info.per_atom is False
    assert target_info.gradients == ["positions"]

    target_info = get_energy_target_info(
        energy_target_config, add_position_gradients=True, add_strain_gradients=True
    )
    assert target_info.quantity == "energy"
    assert target_info.unit == "eV"
    assert target_info.per_atom is False
    assert target_info.gradients == ["positions", "strain"]


def test_layout_scalar(scalar_target_config):
    target_info = get_generic_target_info(scalar_target_config)
    assert target_info.quantity == "scalar"
    assert target_info.unit == ""
    assert target_info.per_atom is False
    assert target_info.gradients == []


def test_layout_cartesian(cartesian_target_config):
    target_info = get_generic_target_info(cartesian_target_config)
    assert target_info.quantity == "dipole"
    assert target_info.unit == "D"
    assert target_info.per_atom is True
    assert target_info.gradients == []


def test_layout_spherical(spherical_target_config):
    target_info = get_generic_target_info(spherical_target_config)
    assert target_info.quantity == "spherical"
    assert target_info.unit == ""
    assert target_info.per_atom is False
    assert target_info.gradients == []
