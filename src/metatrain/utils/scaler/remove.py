from typing import Dict, List

import torch
from metatensor.torch import TensorMap
from metatomic.torch import System


def remove_scale(
    systems: List[System],
    targets: Dict[str, TensorMap],
    scaler: torch.nn.Module,
):
    """
    Scale all targets to a standard deviation of one.

    :param targets: Dictionary containing the targets to be scaled.
    :param scaler: The scaler used to scale the targets.
    """
    return scaler(systems, targets, remove=True)
