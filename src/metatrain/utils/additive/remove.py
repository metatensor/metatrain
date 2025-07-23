import warnings
from typing import Dict, List

import metatensor.torch as mts
import torch
from metatensor.torch import TensorMap
from metatensor.torch.operations._add import _add_block_block
from metatensor.torch.operations._multiply import _multiply_block_constant
from metatomic.torch import System

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
        {
            key: target_info_dict[key]
            for key in targets.keys()
            if key in additive_model.outputs
        },
        is_training=False,  # we don't need any gradients w.r.t. any parameters
    )

    for target_key in additive_contribution.keys():
        # note that we loop over the keys of additive_contribution, not targets,
        # because the targets might contain additional keys (this is for example
        # the case of the composition model, which will only provide outputs
        # for scalar targets

        # make the samples the same so we can use metatensor.torch.subtract
        # we also need to detach the values to avoid backpropagating through the
        # subtraction
        blocks = []
        for block_key, old_block in additive_contribution[target_key].items():
            device = targets[target_key].block(block_key).values.device
            block = mts.TensorBlock(
                values=old_block.values.detach().to(device=device),
                samples=targets[target_key].block(block_key).samples,
                components=[c.to(device=device) for c in old_block.components],
                properties=old_block.properties.to(device=device),
            )
            for gradient_name in targets[target_key].block(block_key).gradients_list():
                gradient = (
                    additive_contribution[target_key]
                    .block(block_key)
                    .gradient(gradient_name)
                )
                block.add_gradient(
                    gradient_name,
                    mts.TensorBlock(
                        values=gradient.values.detach(),
                        samples=targets[target_key]
                        .block(block_key)
                        .gradient(gradient_name)
                        .samples,
                        components=gradient.components,
                        properties=gradient.properties,
                    ),
                )
            blocks.append(block)
        additive_contribution[target_key] = TensorMap(
            keys=additive_contribution[target_key].keys.to(device=device),
            blocks=blocks,
        )
        # Sparse subtract the additive contribution from the appropriate target blocks
        new_target_blocks = []
        for key, block in targets[target_key].items():
            if key in additive_contribution[target_key].keys:
                new_target_blocks.append(
                    _add_block_block(
                        block,
                        _multiply_block_constant(
                            additive_contribution[target_key].block(key),
                            -1.0,
                        ),
                    )
                )

            else:
                new_target_blocks.append(block)

        targets[target_key] = TensorMap(
            keys=targets[target_key].keys,
            blocks=new_target_blocks,
        )

    return targets
