from typing import Tuple

from metatensor.torch.atomistic import ModelCapabilities


def merge_capabilities(
    old_capabilities: ModelCapabilities, new_capabilities: ModelCapabilities
) -> Tuple[ModelCapabilities, ModelCapabilities]:
    """
    Merge the capabilities of a model with the requested capabilities.

    :param old_capabilities: The old capabilities of the model.
    :param new_capabilities: The requested capabilities.

    :return: The merged capabilities and the "novel" capabilities that
        were not present in the old capabilities, but are present in the new
        capabilities. The order is preserved, both in the merged and novel
        capabilities.
    """
    # Check that the length units are the same:
    if old_capabilities.length_unit != new_capabilities.length_unit:
        raise ValueError(
            "The length units of the old and new capabilities are not the same. "
            f"Found `{old_capabilities.length_unit}` and "
            f"`{new_capabilities.length_unit}`."
        )

    # Check that there are no new species:
    for species in new_capabilities.atomic_types:
        if species not in old_capabilities.atomic_types:
            raise ValueError(
                f"The species {species} is not within "
                "the capabilities of the loaded model."
            )

    # Merge the outputs:
    outputs = {}
    for key, value in old_capabilities.outputs.items():
        outputs[key] = value
    for key, value in new_capabilities.outputs.items():
        if key not in outputs:
            outputs[key] = value
        else:
            assert (
                outputs[key].unit == value.unit
            ), f"Output {key} has different units in the old and new capabilities."

    # Find the new outputs:
    new_outputs = {}
    for key, value in new_capabilities.outputs.items():
        if key not in old_capabilities.outputs:
            new_outputs[key] = value

    merged_capabilities = ModelCapabilities(
        length_unit=new_capabilities.length_unit,
        atomic_types=old_capabilities.atomic_types,
        outputs=outputs,
    )

    novel_capabilities = ModelCapabilities(
        length_unit=new_capabilities.length_unit,
        atomic_types=old_capabilities.atomic_types,
        outputs=new_outputs,
    )

    return merged_capabilities, novel_capabilities
