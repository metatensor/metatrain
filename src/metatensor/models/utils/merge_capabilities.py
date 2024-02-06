from typing import Tuple

from metatensor.torch.atomistic import ModelCapabilities


def merge_capabilities(
    old_capabilities: ModelCapabilities, requested_capabilities: ModelCapabilities
) -> Tuple[ModelCapabilities, ModelCapabilities]:
    """
    Merge the capabilities of a model with the requested capabilities.

    :param old_capabilities: The old capabilities of the model.
    :param requested_capabilities: The requested capabilities.

    :return: The merged capabilities and the new capabilities that
        were not present in the old capabilities. The order will
        be preserved.
    """
    # Check that the length units are the same:
    if old_capabilities.length_unit != requested_capabilities.length_unit:
        raise ValueError(
            "The length units of the old and new capabilities are not the same."
        )

    # Check that there are no new species:
    for species in requested_capabilities.species:
        if species not in old_capabilities.species:
            raise ValueError(
                f"The species {species} is not within "
                "the capabilities of the loaded model."
            )

    # Merge the outputs:
    outputs = {}
    for key, value in old_capabilities.outputs.items():
        outputs[key] = value
    for key, value in requested_capabilities.outputs.items():
        if key not in outputs:
            outputs[key] = value
        else:
            assert (
                outputs[key].unit == value.unit
            ), f"Output {key} has different units in the old and new capabilities."

    # Find the new outputs:
    new_outputs = {}
    for key, value in requested_capabilities.outputs.items():
        if key not in old_capabilities.outputs:
            new_outputs[key] = value

    merged_capabilities = ModelCapabilities(
        length_unit=requested_capabilities.length_unit,
        species=old_capabilities.species,
        outputs=outputs,
    )

    new_capabilities = ModelCapabilities(
        length_unit=requested_capabilities.length_unit,
        species=old_capabilities.species,
        outputs=new_outputs,
    )

    return merged_capabilities, new_capabilities
