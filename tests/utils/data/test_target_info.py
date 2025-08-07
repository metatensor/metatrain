import pytest
import torch
from omegaconf import DictConfig

from metatrain.utils.data.target_info import (
    TargetInfo,
    get_energy_target_info,
    get_generic_target_info,
    is_auxiliary_output,
)


@pytest.fixture
def energy_target_config() -> DictConfig:
    return DictConfig(
        {
            "quantity": "energy",
            "unit": "eV",
            "per_atom": False,
            "num_subtargets": 1,
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
            "num_subtargets": 10,
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
            "num_subtargets": 5,
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
            "num_subtargets": 1,
            "type": {
                "spherical": {
                    "irreps": [
                        {"o3_lambda": 0, "o3_sigma": 1},
                        {"o3_lambda": 2, "o3_sigma": 1},
                    ],
                },
            },
        }
    )


def test_layout_energy(energy_target_config):
    target_info = get_energy_target_info(energy_target_config)
    assert target_info.quantity == "energy"
    assert target_info.unit == "eV"
    assert target_info.per_atom is False
    assert target_info.gradients == []
    assert target_info.device == target_info.layout.device

    target_info = get_energy_target_info(
        energy_target_config, add_position_gradients=True
    )
    assert target_info.quantity == "energy"
    assert target_info.unit == "eV"
    assert target_info.per_atom is False
    assert target_info.gradients == ["positions"]
    assert target_info.device == target_info.layout.device

    target_info = get_energy_target_info(
        energy_target_config, add_position_gradients=True, add_strain_gradients=True
    )
    assert target_info.quantity == "energy"
    assert target_info.unit == "eV"
    assert target_info.per_atom is False
    assert target_info.gradients == ["positions", "strain"]
    assert target_info.device == target_info.layout.device


def test_layout_scalar(scalar_target_config):
    target_info = get_generic_target_info(scalar_target_config)
    assert target_info.quantity == "scalar"
    assert target_info.unit == ""
    assert target_info.per_atom is False
    assert target_info.gradients == []
    assert target_info.device == target_info.layout.device


def test_layout_cartesian(cartesian_target_config):
    target_info = get_generic_target_info(cartesian_target_config)
    assert target_info.quantity == "dipole"
    assert target_info.unit == "D"
    assert target_info.per_atom is True
    assert target_info.gradients == []
    assert target_info.device == target_info.layout.device


def test_layout_spherical(spherical_target_config):
    target_info = get_generic_target_info(spherical_target_config)
    assert target_info.quantity == "spherical"
    assert target_info.unit == ""
    assert target_info.per_atom is False
    assert target_info.gradients == []
    assert target_info.device == target_info.layout.device


def test_is_auxiliary_output():
    assert is_auxiliary_output("mtt::aux::energy_uncertainty")
    assert is_auxiliary_output("mtt::aux::energy_last_layer_features")
    assert not is_auxiliary_output("energy")
    assert not is_auxiliary_output("foo")
    assert is_auxiliary_output("mtt::aux::foo")
    assert is_auxiliary_output("features")
    assert is_auxiliary_output("energy_ensemble")
    assert is_auxiliary_output("mtt::aux::energy_ensemble")


def test_is_compatible_with(energy_target_config, spherical_target_config):
    energy_target_info = get_energy_target_info(energy_target_config)
    spherical_target_config = get_generic_target_info(spherical_target_config)
    energy_target_info_with_forces = get_energy_target_info(
        energy_target_config, add_position_gradients=True
    )
    assert energy_target_info.is_compatible_with(energy_target_info)
    assert energy_target_info_with_forces.is_compatible_with(energy_target_info)
    assert not energy_target_info.is_compatible_with(spherical_target_config)
    assert not (
        energy_target_info_with_forces.is_compatible_with(spherical_target_config)
    )


@pytest.mark.parametrize(
    "target_config",
    [
        "energy_target_config",
        "scalar_target_config",
        "cartesian_target_config",
        "spherical_target_config",
    ],
)
def test_instance_torchscript_compatible(target_config, request):
    target_info = get_generic_target_info(request.getfixturevalue(target_config))
    torch.jit.script(target_info)
