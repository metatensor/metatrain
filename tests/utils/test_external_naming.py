from metatensor.models.utils.data.dataset import TargetInfo
from metatensor.models.utils.external_naming import from_external_name, to_external_name


def test_to_external_name():
    """Tests the to_external_name function."""

    quantities = {
        "energy": TargetInfo(quantity="energy"),
        "mtm::free_energy": TargetInfo(quantity="energy"),
        "mtm::foo": TargetInfo(quantity="bar"),
    }

    assert to_external_name("energy_positions_gradients", quantities) == "forces"
    assert (
        to_external_name("mtm::free_energy_positions_gradients", quantities)
        == "forces[mtm::free_energy]"
    )
    assert (
        to_external_name("mtm::foo_positions_gradients", quantities)
        == "mtm::foo_positions_gradients"
    )
    assert to_external_name("energy_strain_gradients", quantities) == "virial"
    assert (
        to_external_name("mtm::free_energy_strain_gradients", quantities)
        == "virial[mtm::free_energy]"
    )
    assert (
        to_external_name("mtm::foo_strain_gradients", quantities)
        == "mtm::foo_strain_gradients"
    )
    assert to_external_name("energy", quantities) == "energy"
    assert to_external_name("mtm::free_energy", quantities) == "mtm::free_energy"
    assert to_external_name("mtm::foo", quantities) == "mtm::foo"


def test_from_external_name():
    """Tests the from_external_name function."""

    assert from_external_name("forces") == "energy_positions_gradients"
    assert (
        from_external_name("forces[mtm::free_energy]")
        == "mtm::free_energy_positions_gradients"
    )
    assert (
        from_external_name("mtm::foo_positions_gradients")
        == "mtm::foo_positions_gradients"
    )
    assert from_external_name("virial") == "energy_strain_gradients"
    assert (
        from_external_name("virial[mtm::free_energy]")
        == "mtm::free_energy_strain_gradients"
    )
    assert (
        from_external_name("mtm::foo_strain_gradients") == "mtm::foo_strain_gradients"
    )
    assert from_external_name("energy") == "energy"
    assert from_external_name("mtm::free_energy") == "mtm::free_energy"
    assert from_external_name("mtm::foo") == "mtm::foo"
