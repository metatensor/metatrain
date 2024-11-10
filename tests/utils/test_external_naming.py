from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.external_naming import to_external_name, to_internal_name


def test_to_external_name():
    """Tests the to_external_name function."""

    quantities = {
        "energy": get_energy_target_info({"quantity": "energy", "unit": "eV"}),
        "mtt::free_energy": get_energy_target_info(
            {"quantity": "energy", "unit": "eV"}
        ),
        "mtt::foo": get_energy_target_info({"quantity": "energy", "unit": "eV"}),
    }

    assert to_external_name("energy_positions_gradients", quantities) == "forces"
    assert (
        to_external_name("mtt::free_energy_positions_gradients", quantities)
        == "forces[mtt::free_energy]"
    )
    assert (
        to_external_name("mtt::foo_positions_gradients", quantities)
        == "mtt::foo_positions_gradients"
    )
    assert to_external_name("energy_strain_gradients", quantities) == "virial"
    assert (
        to_external_name("mtt::free_energy_strain_gradients", quantities)
        == "virial[mtt::free_energy]"
    )
    assert (
        to_external_name("mtt::foo_strain_gradients", quantities)
        == "mtt::foo_strain_gradients"
    )
    assert to_external_name("energy", quantities) == "energy"
    assert to_external_name("mtt::free_energy", quantities) == "mtt::free_energy"
    assert to_external_name("mtt::foo", quantities) == "mtt::foo"


def test_to_internal_name():
    """Tests the to_internal_name function."""

    assert to_internal_name("forces") == "energy_positions_gradients"
    assert (
        to_internal_name("forces[mtt::free_energy]")
        == "mtt::free_energy_positions_gradients"
    )
    assert (
        to_internal_name("mtt::foo_positions_gradients")
        == "mtt::foo_positions_gradients"
    )
    assert to_internal_name("virial") == "energy_strain_gradients"
    assert (
        to_internal_name("virial[mtt::free_energy]")
        == "mtt::free_energy_strain_gradients"
    )
    assert to_internal_name("mtt::foo_strain_gradients") == "mtt::foo_strain_gradients"
    assert to_internal_name("energy") == "energy"
    assert to_internal_name("mtt::free_energy") == "mtt::free_energy"
    assert to_internal_name("mtt::foo") == "mtt::foo"
