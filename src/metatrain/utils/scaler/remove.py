from typing import Callable, Dict, List, Tuple

import torch
from metatensor.torch import TensorMap
from metatomic.torch import System

from .scaler import Scaler


def remove_scale(
    systems: List[System],
    targets: Dict[str, TensorMap],
    scaler: torch.nn.Module,
    use_global_scales: bool,
    use_property_scales: bool,
) -> Dict[str, TensorMap]:
    """
    Scale all targets to a standard deviation of one.

    :param systems: List of systems corresponding to the targets.
    :param targets: Dictionary containing the targets to be scaled.
    :param scaler: The scaler used to scale the targets.
    :return: The scaled targets.
    """
    return scaler(
        systems,
        targets,
        remove=True,
        use_global_scales=use_global_scales,
        use_property_scales=use_property_scales,
    )


def get_remove_scale_transform(
    scaler: Scaler, use_global_scales: bool, use_property_scales: bool
) -> Callable:
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
        new_targets = remove_scale(
            systems, targets, scaler, use_global_scales, use_property_scales
        )
        return systems, new_targets, extra

    return transform


def get_remove_scale_transform_with_logging(
    scaler: torch.nn.Module,
    use_global_scales: bool,
    use_property_scales: bool,
    rescale_prediction_properties: bool,
    logging: torch.nn.Module,
) -> List[Callable]:
    """Handles the logic of which remove scale transforms to use based on
    the scaling hyperparameters, with logging for clarity."""
    if use_global_scales:
        if use_property_scales:
            if rescale_prediction_properties:
                logging.info(
                    "Training with global and per-property scaling. Prediction"
                    "  properties will be rescaled before loss calculation."
                )
                remove_scale_transform = [
                    get_remove_scale_transform(
                        scaler,
                        use_global_scales=True,
                        use_property_scales=False,  # predictions rescaled
                    )
                ]
            else:
                logging.info("Training with global and per-property scaling.")
                remove_scale_transform = [
                    get_remove_scale_transform(
                        scaler,
                        use_global_scales=True,
                        use_property_scales=True,  # targets scaled
                    )
                ]
        else:
            logging.info("Training with global scaling.")
            remove_scale_transform = [
                get_remove_scale_transform(
                    scaler,
                    use_global_scales=True,
                    use_property_scales=False,  # no per-property scaling
                )
            ]
    else:
        if use_property_scales:
            if rescale_prediction_properties:
                logging.info("Training with per-property scaling.")
                remove_scale_transform = [
                    get_remove_scale_transform(
                        scaler,
                        use_global_scales=False,
                        use_property_scales=False,  # predictions rescaled
                    )
                ]
            else:
                logging.info(
                    "Training with per-property scaling. Prediction"
                    "  properties will be rescaled before loss calculation."
                )
                remove_scale_transform = [
                    get_remove_scale_transform(
                        scaler,
                        use_global_scales=False,
                        use_property_scales=True,  # targets rescaled
                    )
                ]
        else:
            logging.info("No target scaling.")
            remove_scale_transform = []  # no scaling

    return remove_scale_transform
