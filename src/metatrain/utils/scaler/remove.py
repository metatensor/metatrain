from typing import Callable, Dict, List, Tuple

import metatensor.torch as mts
import torch
from metatensor.torch import TensorBlock, TensorMap
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
        per_property_scaled = scaler(
            systems,
            targets,
            remove=True,
            use_per_target_scales=False,
            use_per_property_scales=True,
        )

        def NaNs_to_1(tensormap: TensorMap) -> TensorMap:
            """If the targets with the removed scales contain 0s,
            computing the scales by dividing will give NaNs.
            This function replaces those NaNs with 1s, to avoid
            issues during training
            (the true scale in this case does not really matter)"""
            new_blocks = []
            for block in tensormap.blocks():
                values = block.values
                values[torch.isnan(values)] = 1.0
                new_blocks.append(
                    TensorBlock(
                        values=values,
                        samples=block.samples,
                        components=block.components,
                        properties=block.properties,
                    )
                )
            return TensorMap(tensormap.keys, new_blocks)

        for key in targets.keys():
            scales = mts.divide(targets[key], new_targets[key])
            per_property_scales = mts.divide(targets[key], per_property_scaled[key])

            extra[f"mtt::aux::scales::{key}"] = NaNs_to_1(scales)
            extra[f"mtt::aux::per-property-scales::{key}"] = NaNs_to_1(
                per_property_scales
            )

        return systems, new_targets, extra

    return transform
