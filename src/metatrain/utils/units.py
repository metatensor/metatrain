from typing import Tuple

from metatomic.torch import unit_conversion_factor


def get_gradient_units(base_unit: str, gradient_name: str, length_unit: str) -> str:
    """
    Get the gradient units based on the unit of the base quantity.

    For example, if the base unit is "<unit>" and the gradient name is
    "positions", the gradient unit will be "<unit>/<length_unit>".

    :param base_unit: The unit of the base quantity.
    :param gradient_name: The name of the gradient.
    :param length_unit: The unit of lengths.

    :return: The unit of the gradient.
    """
    if base_unit == "":
        return ""  # unknown unit for base quantity -> unknown unit for gradient
    if length_unit.lower() in ["angstrom", "å", "ångstrom"]:
        length_unit = "A"  # prettier
    if gradient_name == "positions":
        return base_unit + "/" + length_unit
    elif gradient_name == "strain":
        return base_unit  # strain is dimensionless
    else:
        raise ValueError(f"Unknown gradient name: {gradient_name}")


def ev_to_mev(value: float, unit: str) -> Tuple[float, str]:
    """
    If the `unit` starts with eV, converts the `value` and its
    corresponding `unit` to meV. Otherwise, returns the input.

    :param value: The value (potentially in eV or a derived quantity of eV).
    :param unit: The unit of the value.
    :return: If the `value` is in meV (or a derived quantity), the value and
        the corresponding unit where eV is converted to meV. Otherwise, the input.
    """
    if unit.startswith("eV") or unit.startswith("ev"):
        return value * 1000.0, (
            unit.replace("eV", "meV")
            if unit.startswith("eV")
            else unit.replace("ev", "mev")
        )
    else:
        return value, unit


def units_are_equivalent(unit: str, other_unit: str) -> bool:
    """Check whether two unit expressions represent the same numerical unit.

    Equivalent units have a conversion factor of exactly one, so their data can
    be combined without rescaling. Unknown units (empty strings) are only
    equivalent to other unknown units.

    :param unit: The first unit expression.
    :param other_unit: The second unit expression.
    :return: Whether the units are equivalent without rescaling.
    """
    if unit == other_unit:
        return True
    if unit == "" or other_unit == "":
        return False

    try:
        return unit_conversion_factor(unit, other_unit) == 1.0
    except ValueError:
        return False
