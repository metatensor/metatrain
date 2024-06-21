import pytest

from metatrain.utils.units import ev_to_mev, get_gradient_units


def test_get_gradient_units():
    """Tests the get_gradient_units function."""
    # Test the case where the base unit is empty
    assert get_gradient_units("", "positions", "angstrom") == ""
    # Test the case where the length unit is angstrom
    assert get_gradient_units("unit", "positions", "angstrom") == "unit/Å"
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
