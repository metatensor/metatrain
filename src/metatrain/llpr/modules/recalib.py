from typing import List, Optional

import torch.nn as nn


def apply_ensemble_training_strategy(
    model: nn.Module, train_all_parameters: bool,
) -> nn.Module:
    """
    Apply the user-specified ensemble training strategy to the LLPR-wrapped
    model. This function modifies the model in place based on the provided
    trainable parameters.

    :param model: LLPR-wrapped model to be recalibrated.
    :param train_all_parameters: Whether to train all parameters or only the LLPR
        ensemble layers.
    :return: the model with updated trainable parameters.
    """

    # Start by making all parameters trainable
    for param in model.parameters():
        param.requires_grad = True

    if not train_all_parameters:
        # Freeze all parameters of the base model
        for param in model.model.parameters():
            param.requires_grad = False

    return model
