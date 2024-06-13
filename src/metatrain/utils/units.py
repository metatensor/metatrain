from typing import Tuple


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
    if length_unit == "angstrom":
        length_unit = "Å"  # prettier
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
