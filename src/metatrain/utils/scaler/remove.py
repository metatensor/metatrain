from typing import Callable, Dict, List, Tuple

import torch
from metatensor.torch import TensorMap
from metatomic.torch import System

from .scaler import Scaler


def remove_scale(
    systems: List[System],
    targets: Dict[str, TensorMap],
    scaler: torch.nn.Module,
) -> Dict[str, TensorMap]:
    """
    Scale all targets to a standard deviation of one.

    :param systems: List of systems corresponding to the targets.
    :param targets: Dictionary containing the targets to be scaled.
    :param scaler: The scaler used to scale the targets.
    :return: The scaled targets.
    """
    return scaler(systems, targets, remove=True)


def get_remove_scale_transform(scaler: Scaler) -> Callable:
    """
    Remove the scaling from the targets using the provided scaler.

    :param scaler: The scaler used to scale the targets.
    :return: A function that removes the scaling from the targets.
    """

    def transform(
        systems: List[System],
        targets: Dict[str, TensorMap],
        extra: Dict[str, TensorMap],
    ) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
        """
        :param systems: List of systems.
        :param targets: Dictionary containing the targets corresponding to the systems.
        :param extra: Dictionary containing any extra data.
        :return: The systems, updated targets and extra data.
        """
        new_targets = remove_scale(systems, targets, scaler)
        return systems, new_targets, extra

    return transform
