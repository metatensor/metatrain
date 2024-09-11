from typing import Dict, List

import torch
from metatensor.torch import TensorMap
from metatensor.torch.atomistic import ModelOutput, System


def remove_additive(
    systems: List[System],
    targets: Dict[str, TensorMap],
    composition_model: torch.nn.Module,
):
    """Remove an additive contribution from the training targets.

    The targets are changed in place.

    :param systems: List of systems.
    :param targets: Dictionary containing the targets corresponding to the systems.
    :param additive_model: The model used to calculate the additive
        contribution to be removed.
    """
    output_options = {}
    for target_key in targets:
        output_options[target_key] = ModelOutput(per_atom=False)

    composition_targets = composition_model(systems, output_options)
    for target_key in targets:
        targets[target_key].block().values[:] -= (
            composition_targets[target_key].block().values
        )
