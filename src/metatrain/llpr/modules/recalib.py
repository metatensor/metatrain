from typing import List, Optional

import torch.nn as nn


def apply_ensemble_training_strategy(
    model: nn.Module, target: str, trainable_parameters: Optional[List[str]]
) -> nn.Module:
    """
    Apply the user-specified ensemble training strategy to the LLPR-wrapped
    model. This function modifies the model in place based on the provided
    trainable parameters.

    :param model: LLPR-wrapped model to be recalibrated.
    :param target: target property for which ensemble training is performed.
    :param trainable_parameters: Optional list of parameter names to train.
        If None, all parameters are trained.
        If an empty list or a list of specific names, only those parameters are trained.
    :return: the model with updated trainable parameters.
    """

    # Start by freezing all parameters
    for param in model.model.parameters():
        param.requires_grad = False
    for module in model.llpr_ensemble_layers.values():
        for param in module.parameters():
            param.requires_grad = False

    if trainable_parameters is None:
        # Train all parameters (both wrapped model and ensemble layers)
        for param in model.model.parameters():
            param.requires_grad = True
        for module in model.llpr_ensemble_layers.values():
            for param in module.parameters():
                param.requires_grad = True
    else:
        # Train only the specified parameters
        # First, always enable ensemble layer training
        for module in model.llpr_ensemble_layers.values():
            for param in module.parameters():
                param.requires_grad = True

        # Then enable the specified wrapped model parameters
        if len(trainable_parameters) > 0:
            remaining_params = trainable_parameters.copy()
            for name, param in model.model.named_parameters():
                if name in remaining_params:
                    param.requires_grad = True
                    remaining_params.remove(name)
            if len(remaining_params) > 0:
                raise RuntimeError(
                    f"Not all specified parameters were found in the wrapped model!\n"
                    f"Remaining params: {remaining_params}"
                )

    return model
