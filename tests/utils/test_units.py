import pytest

from metatrain.utils.units import ev_to_mev, get_gradient_units, units_are_equivalent


def test_get_gradient_units():
    """Tests the get_gradient_units function."""
    # Test the case where the base unit is empty
    assert get_gradient_units("", "positions", "angstrom") == ""
    # Test the case where the length unit is angstrom
    for length_unit in ["angstrom", "å", "ångstrom", "Ångstrom", "Å"]:
        assert get_gradient_units("unit", "positions", length_unit) == "unit/A"
    # Test the case where the gradient name is strain
    assert get_gradient_units("unit", "strain", "angstrom") == "unit"
    # Test the case where the gradient name is unknown
    with pytest.raises(ValueError):
        get_gradient_units("unit", "unknown", "angstrom")


def test_ev_to_mev():
    """Tests the ev_to_mev function."""
    # Test the case where the unit is not eV
    assert ev_to_mev(1.0, "unit") == (1.0, "unit")
    # Test the case where the unit is eV
    assert ev_to_mev(1.0, "eV") == (1000.0, "meV")
    # Test the case where the unit is eV with a different case
    assert ev_to_mev(0.2, "ev") == (200.0, "mev")
    # Test the case where the unit is a derived unit of eV
    assert ev_to_mev(1.0, "eV/unit") == (1000.0, "meV/unit")


@pytest.mark.parametrize(
    "unit, other_unit, equivalent",
    [
        pytest.param("eV", "eV", True, id="identical-units"),
        pytest.param("", "", True, id="both-units-empty"),
        pytest.param("", "eV", False, id="one-unit-empty"),
        pytest.param("A", "angstrom", True, id="length-alias"),
        pytest.param("eV/A", "eV/angstrom", True, id="derived-unit-alias"),
        pytest.param("eV/A^3", "eV/angstrom^3", True, id="powered-unit-alias"),
        pytest.param("eV", "meV", False, id="different-energy-scales"),
        pytest.param("angstrom", "nm", False, id="different-length-scales"),
        pytest.param("eV", "angstrom", False, id="different-dimensions"),
        pytest.param("invalid-unit", "eV", False, id="invalid-unit"),
        pytest.param("eV/(", "eV", False, id="malformed-unit"),
    ],
)
def test_units_are_equivalent(unit: str, other_unit: str, equivalent: bool):
    assert units_are_equivalent(unit, other_unit) is equivalent
    assert units_are_equivalent(other_unit, unit) is equivalent
