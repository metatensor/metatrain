from typing import Dict, List

import metatensor.torch
import torch
from metatensor.torch import TensorMap
from metatensor.torch.atomistic import System

from ..data import TargetInfoDict
from ..evaluate_model import evaluate_model


def remove_additive(
    systems: List[System],
    targets: Dict[str, TensorMap],
    additive_model: torch.nn.Module,
    target_info_dict: TargetInfoDict,
):
    """Remove an additive contribution from the training targets.

    :param systems: List of systems.
    :param targets: Dictionary containing the targets corresponding to the systems.
    :param additive_model: The model used to calculate the additive
        contribution to be removed.
    :param targets_dict: Dictionary containing information about the targets.
    """
    additive_contribution = evaluate_model(
        additive_model,
        systems,
        TargetInfoDict(**{key: target_info_dict[key] for key in targets.keys()}),
        is_training=False,
    )

    for target_key in targets:
        targets[target_key] = metatensor.torch.subtract(
            targets[target_key], additive_contribution[target_key]
        )

    return targets
