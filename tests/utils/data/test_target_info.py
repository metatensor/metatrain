from typing import Literal

import pytest
import torch
from omegaconf import DictConfig

from metatrain.utils.data.target_info import (
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
            "description": "Total potential energy of the system",
            "sample_kind": "system",
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
            "sample_kind": "system",
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
            "sample_kind": "atom",
            "num_subtargets": 5,
            "type": {
                "Cartesian": {
                    "rank": 1,
                }
            },
        }
    )


@pytest.fixture(params=[None, "coupled"])
def spherical_product(request) -> Literal[None, "coupled"]:
    return request.param


@pytest.fixture
def spherical_target_config(spherical_product) -> DictConfig:
    return DictConfig(
        {
            "quantity": "spherical",
            "unit": "",
            "sample_kind": "system",
            "num_subtargets": 1,
            "type": {
                "spherical": {
                    "product": spherical_product,
                    "irreps": [
                        {"o3_lambda": 0, "o3_sigma": 1},
                        {"o3_lambda": 1, "o3_sigma": 1},
                        {"o3_lambda": 2, "o3_sigma": 1},
                    ],
                },
            },
        }
    )


@pytest.fixture
def spherical_atomicbasis_target_config(spherical_product) -> DictConfig:
    return DictConfig(
        {
            "quantity": "spherical_atomicbasis",
            "unit": "",
            "sample_kind": "atom",
            "num_subtargets": 1,
            "type": {
                "spherical": {
                    "product": spherical_product,
                    "irreps": {
                        1: [{"o3_lambda": 0, "o3_sigma": 1}],
                        6: [
                            {"o3_lambda": 0, "o3_sigma": 1},
                            {"o3_lambda": 1, "o3_sigma": 1},
                            {"o3_lambda": 2, "o3_sigma": 1},
                        ],
                    },
                },
            },
        }
    )


def test_layout_energy(energy_target_config):
    target_info = get_energy_target_info("energy", energy_target_config)
    assert target_info.quantity == "energy"
    assert target_info.unit == "eV"
    assert target_info.sample_kind == "system"
    assert target_info.description == "Total potential energy of the system"
    assert target_info.gradients == []
    assert target_info.device == target_info.layout.device

    target_info = get_energy_target_info(
        "energy", energy_target_config, add_position_gradients=True
    )
    assert target_info.quantity == "energy"
    assert target_info.unit == "eV"
    assert target_info.sample_kind == "system"
    assert target_info.gradients == ["positions"]
    assert target_info.device == target_info.layout.device

    target_info = get_energy_target_info(
        "energy",
        energy_target_config,
        add_position_gradients=True,
        add_strain_gradients=True,
    )
    assert target_info.quantity == "energy"
    assert target_info.unit == "eV"
    assert target_info.sample_kind == "system"
    assert target_info.gradients == ["positions", "strain"]
    assert target_info.device == target_info.layout.device

    repr_str = repr(target_info)
    reprs = (
        f"TargetInfo(layout={target_info.layout}, quantity='{target_info.quantity}', "
        f"unit='{target_info.unit}', description='{target_info.description}')"
    )
    assert reprs == repr_str

    # Check that TargetInfo correctly identifies the type of target.
    assert target_info.is_scalar
    assert not target_info.is_cartesian
    assert not target_info.is_spherical
    assert not target_info.is_atomic_basis


def test_layout_scalar(scalar_target_config):
    target_info = get_generic_target_info("scalar", scalar_target_config)
    assert target_info.quantity == "scalar"
    assert target_info.unit == ""
    assert target_info.sample_kind == "system"
    assert target_info.gradients == []
    assert target_info.device == target_info.layout.device

    # Check that TargetInfo correctly identifies the type of target.
    assert target_info.is_scalar
    assert not target_info.is_cartesian
    assert not target_info.is_spherical
    assert not target_info.is_atomic_basis


def test_layout_cartesian(cartesian_target_config):
    target_info = get_generic_target_info("cartesian", cartesian_target_config)
    assert target_info.quantity == "dipole"
    assert target_info.unit == "D"
    assert target_info.sample_kind == "atom"
    assert target_info.gradients == []
    assert target_info.device == target_info.layout.device

    # Check that TargetInfo correctly identifies the type of target.
    assert not target_info.is_scalar
    assert target_info.is_cartesian
    assert not target_info.is_spherical
    assert not target_info.is_atomic_basis


def test_layout_spherical(spherical_target_config, spherical_product):
    target_info = get_generic_target_info("spherical", spherical_target_config)
    assert target_info.quantity == "spherical"
    assert target_info.unit == ""
    assert target_info.sample_kind == "system"
    assert target_info.gradients == []
    assert target_info.device == target_info.layout.device

    # Check that TargetInfo correctly identifies the type of target.
    assert not target_info.is_scalar
    assert not target_info.is_cartesian
    assert target_info.is_spherical
    assert not target_info.is_atomic_basis

    if spherical_product is None:
        assert len(target_info.layout.blocks()) == 3
    elif spherical_product == "coupled":
        assert len(target_info.layout.blocks()) == 5


def test_layout_spherical_atomicbasis(
    spherical_atomicbasis_target_config, spherical_product
):
    target_info = get_generic_target_info(
        "spherical_atomicbasis", spherical_atomicbasis_target_config
    )
    assert target_info.quantity == "spherical_atomicbasis"
    assert target_info.unit == ""
    assert target_info.sample_kind == "atom"
    assert target_info.gradients == []
    assert target_info.device == target_info.layout.device

    # Check that TargetInfo correctly identifies the type of target.
    assert not target_info.is_scalar
    assert not target_info.is_cartesian
    assert target_info.is_spherical
    assert target_info.is_atomic_basis

    if spherical_product is None:
        assert len(target_info.layout.blocks()) == 4
    elif spherical_product == "coupled":
        assert len(target_info.layout.blocks()) == 6


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
    energy_target_info = get_energy_target_info("energy", energy_target_config)
    spherical_target_config = get_generic_target_info(
        "spherical", spherical_target_config
    )
    energy_target_info_with_forces = get_energy_target_info(
        "energy", energy_target_config, add_position_gradients=True
    )
    assert energy_target_info.is_compatible_with(energy_target_info)
    assert energy_target_info_with_forces.is_compatible_with(energy_target_info)
    assert not energy_target_info.is_compatible_with(spherical_target_config)
    assert not (
        energy_target_info_with_forces.is_compatible_with(spherical_target_config)
    )


def _test_instance_torchscript_compatible(target_config):
    target_info = get_generic_target_info("target_name", target_config)
    torch.jit.script(target_info)


# Check that all possible target info instances are compatible with TorchScript.
def test_torchscript_energy(energy_target_config):
    _test_instance_torchscript_compatible(energy_target_config)


def test_torchscript_scalar(scalar_target_config):
    _test_instance_torchscript_compatible(scalar_target_config)


def test_torchscript_cartesian(cartesian_target_config):
    _test_instance_torchscript_compatible(cartesian_target_config)


def test_torchscript_spherical(spherical_target_config):
    _test_instance_torchscript_compatible(spherical_target_config)


def test_torchscript_spherical_atomicbasis(spherical_atomicbasis_target_config):
    _test_instance_torchscript_compatible(spherical_atomicbasis_target_config)


def test_invalid_unit():
    conf = DictConfig(
        {
            "quantity": "energy",
            "unit": "fooo",
            "description": "Total potential energy of the system",
            "sample_kind": "system",
            "num_subtargets": 1,
            "type": "scalar",
        }
    )

    with pytest.raises(ValueError, match="fooo"):
        get_generic_target_info("energy", conf)


def warn_unknown_quantity():
    conf = DictConfig(
        {
            "quantity": "fooo",
            "unit": "fooo",
            "description": "Some description",
            "sample_kind": "system",
            "num_subtargets": 1,
            "type": "scalar",
        }
    )

    with pytest.warns(UserWarning, match="unknown quantity 'fooo'"):
        get_generic_target_info("some_target", conf)


@pytest.mark.parametrize(
    "target_name",
    ["scalar/variant1", "mtt::scalar", "mtt::scalar/variant1"],
)
def test_layout_scalar_with_variant(scalar_target_config, target_name):
    """Test that scalar targets with variant names and/or mtt:: prefix work correctly.

    The '/' character is not accepted in Labels, so the variant part should be
    removed when creating the properties labels. The 'mtt::' prefix should also be
    removed.
    """
    target_info = get_generic_target_info(target_name, scalar_target_config)
    assert target_info.quantity == "scalar"
    assert target_info.unit == ""
    assert target_info.sample_kind == "system"
    assert target_info.gradients == []

    # Check that the properties labels were created correctly without the variant part
    # and mtt:: prefix. The properties label should be "scalar" in all cases.
    block = target_info.layout.block()
    assert block.properties.names == ["scalar"]


@pytest.mark.parametrize(
    "target_name",
    ["dipole/variant1", "mtt::dipole", "mtt::dipole/variant1"],
)
def test_layout_cartesian_with_variant(cartesian_target_config, target_name):
    target_info = get_generic_target_info(target_name, cartesian_target_config)
    assert target_info.quantity == "dipole"
    assert target_info.unit == "D"
    assert target_info.sample_kind == "atom"
    assert target_info.gradients == []

    # Check that the properties labels were created correctly without the variant part
    # and mtt:: prefix. The properties label should be "dipole" in all cases.
    block = target_info.layout.block()
    assert block.properties.names == ["dipole"]


@pytest.mark.parametrize(
    "target_name",
    ["spherical/variant1", "mtt::spherical", "mtt::spherical/variant1"],
)
def test_layout_spherical_with_variant(
    spherical_target_config, target_name, spherical_product
):
    target_info = get_generic_target_info(target_name, spherical_target_config)
    assert target_info.quantity == "spherical"
    assert target_info.unit == ""
    assert target_info.sample_kind == "system"
    assert target_info.gradients == []

    # Check that the properties labels were created correctly without the variant part
    # and mtt:: prefix. The properties label should be the same in all cases.
    for block in target_info.layout.blocks():
        if spherical_product == "coupled":
            assert block.properties.names == ["l_1", "l_2", "n_1", "n_2"]
        else:
            assert block.properties.names == ["n"]


@pytest.mark.parametrize(
    "target_name",
    [
        "spherical_atomicbasis/variant1",
        "mtt::spherical_atomicbasis",
        "mtt::spherical_atomicbasis/variant1",
    ],
)
def test_layout_spherical_atomicbasis_with_variant(
    spherical_atomicbasis_target_config, target_name, spherical_product
):
    target_info = get_generic_target_info(
        target_name, spherical_atomicbasis_target_config
    )
    assert target_info.quantity == "spherical_atomicbasis"
    assert target_info.unit == ""
    assert target_info.sample_kind == "atom"
    assert target_info.gradients == []

    # Check that the properties labels were created correctly without the variant part
    # and mtt:: prefix. The properties label should be the same in all cases.
    for block in target_info.layout.blocks():
        if spherical_product == "coupled":
            assert block.properties.names == ["l_1", "l_2", "n_1", "n_2"]
        else:
            assert block.properties.names == ["n"]
