import functools
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
    Remove global scales from the targets using the provided scaler. It leaves the
    per-property scales unchanged.

    :param systems: List of systems corresponding to the targets.
    :param targets: Dictionary containing the targets to be scaled.
    :param scaler: The scaler used to scale the targets.
    :return: The scaled targets.
    """
    return scaler(
        systems,
        targets,
        remove=True,
        use_per_target_scales=True,
        use_per_property_scales=False,
    )


def _remove_scale_transform_impl(
    scaler: Scaler,
    systems: List[System],
    targets: Dict[str, TensorMap],
    extra: Dict[str, TensorMap],
) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
    new_targets = remove_scale(systems, targets, scaler)
    return systems, new_targets, extra


def get_remove_scale_transform(scaler: Scaler) -> Callable:
    """
    Remove the scaling from the targets using the provided scaler.

    :param scaler: The scaler used to scale the targets.
    :return: A function that removes the scaling from the targets.
    """
    return functools.partial(_remove_scale_transform_impl, scaler)
