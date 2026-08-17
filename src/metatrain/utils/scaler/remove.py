from typing import Callable, Dict, List, Tuple

import torch
from metatensor.torch import TensorBlock, TensorMap
from metatomic.torch import System

from metatrain.scaler import Scaler


def removed_scale_name(target_name: str) -> str:
    """Return the ``extra_data`` key recording a target's removed scale.

    The map stored there is shaped like the target itself (NaN pattern
    included) and holds, per entry, the reciprocal of the per-target scale
    that :py:func:`get_remove_scale_transform` divided out. Any loss that is not
    a plain elementwise one on the target has to undo the scaling through this
    record, because the transform is the only place that knows what was applied,
    and what it applies is a *diagonal* factor, not a scalar: a per-atom target
    is scaled per atomic type. That includes losses which would be invariant
    under a scalar (e.g. the quadratic density losses, whose metric would
    otherwise become ``D^-1 M D^-1``) as well as any loss that is not
    homogeneous in the target at all.

    :param target_name: Name of the target.
    :return: The ``extra_data`` key.
    """
    return f"{target_name}_removed_scale"


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
    return scaler.apply_scales(
        systems,
        targets,
        remove=True,
        use_per_target_scales=True,
        use_per_property_scales=False,
    )


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
        # Record what was removed, under removed_scale_name: a ones map with
        # each target's own NaN pattern (``0 * x + 1`` keeps NaN), sent through
        # the *same* scale removal, holds the reciprocal factor per entry --
        # exact whatever the scale structure (per block, per atomic type),
        # because it takes the identical code path as the target.
        ones = {
            name: TensorMap(
                tensor.keys,
                [
                    TensorBlock(
                        values=0.0 * tensor.block(key).values + 1.0,
                        samples=tensor.block(key).samples,
                        components=tensor.block(key).components,
                        properties=tensor.block(key).properties,
                    )
                    for key in tensor.keys
                ],
            )
            for name, tensor in targets.items()
        }
        removed = remove_scale(systems, ones, scaler)
        for name, tensor in removed.items():
            extra[removed_scale_name(name)] = tensor
        return systems, new_targets, extra

    return transform
