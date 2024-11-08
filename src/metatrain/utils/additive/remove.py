import warnings
from typing import Dict, List

import metatensor.torch
import torch
from metatensor.torch import TensorMap
from metatensor.torch.atomistic import System

from ..data import TargetInfo
from ..evaluate_model import evaluate_model


def remove_additive(
    systems: List[System],
    targets: Dict[str, TensorMap],
    additive_model: torch.nn.Module,
    target_info_dict: Dict[str, TargetInfo],
):
    """Remove an additive contribution from the training targets.

    :param systems: List of systems.
    :param targets: Dictionary containing the targets corresponding to the systems.
    :param additive_model: The model used to calculate the additive
        contribution to be removed.
    :param targets_dict: Dictionary containing information about the targets.
    """
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message=(
            "GRADIENT WARNING: element 0 of tensors does not "
            "require grad and does not have a grad_fn"
        ),
    )
    additive_contribution = evaluate_model(
        additive_model,
        systems,
        {key: target_info_dict[key] for key in targets.keys()},
        is_training=False,  # we don't need any gradients w.r.t. any parameters
    )

    for target_key in targets:
        # make the samples the same so we can use metatensor.torch.subtract
        # we also need to detach the values to avoid backpropagating through the
        # subtraction
        block = metatensor.torch.TensorBlock(
            values=additive_contribution[target_key].block().values.detach(),
            samples=targets[target_key].block().samples,
            components=additive_contribution[target_key].block().components,
            properties=additive_contribution[target_key].block().properties,
        )
        for gradient_name, gradient in (
            additive_contribution[target_key].block().gradients()
        ):
            block.add_gradient(
                gradient_name,
                metatensor.torch.TensorBlock(
                    values=gradient.values.detach(),
                    samples=targets[target_key].block().gradient(gradient_name).samples,
                    components=gradient.components,
                    properties=gradient.properties,
                ),
            )
        additive_contribution[target_key] = TensorMap(
            keys=targets[target_key].keys,
            blocks=[
                block,
            ],
        )
        # subtract the additive contribution from the target
        targets[target_key] = metatensor.torch.subtract(
            targets[target_key], additive_contribution[target_key]
        )

    return targets
