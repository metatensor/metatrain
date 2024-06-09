from typing import Dict, Union

from metatensor.torch.atomistic import ModelOutput


def to_external_name(
    internal_name: str, quantities: Union[Dict[str, ModelOutput]]
) -> str:
    """Converts internal names to external names.

    Very often, the "common" names for quantities are different from the
    internal names used in the code. Two important examples are forces and
    virials, which are referred to as energy_positions_gradients and
    energy_strain_gradients, respectively, in the code. This function
    converts an internal name to an external name.

    :param internal_name: An internal name to convert.
    :param quantities: A dictionary of physical quantities, either as
        :py:class:`TargetInfo` objects or as :py:class:`ModelOutput` objects.

    :return: The name for external use.
    """

    if internal_name.endswith("_positions_gradients"):
        base_name = internal_name.replace("_positions_gradients", "")
        if quantities[base_name].quantity == "energy":
            if base_name == "energy":  # we treat "energy" as a special case
                external_name = "forces"
            else:
                external_name = f"forces[{base_name}]"
        else:
            external_name = internal_name
    elif internal_name.endswith("_strain_gradients"):
        base_name = internal_name.replace("_strain_gradients", "")
        if quantities[base_name].quantity == "energy":
            if base_name == "energy":
                external_name = "virial"
            else:
                external_name = f"virial[{base_name}]"
        else:
            external_name = internal_name
    else:
        external_name = internal_name

    return external_name


def to_internal_name(external_name: str) -> str:
    """Converts an external names to internal names.

    This function is the inverse of :func:`to_external_names`.

    :param external_names: A list of names to convert.

    :return: The list of names for internal use.
    """

    if external_name == "forces":
        internal_name = "energy_positions_gradients"
    elif external_name.startswith("forces[") and external_name.endswith("]"):
        base_name = external_name[7:-1]
        internal_name = f"{base_name}_positions_gradients"
    elif external_name == "virial":
        internal_name = "energy_strain_gradients"
    elif external_name.startswith("virial[") and external_name.endswith("]"):
        base_name = external_name[7:-1]
        internal_name = f"{base_name}_strain_gradients"
    else:
        internal_name = external_name

    return internal_name
