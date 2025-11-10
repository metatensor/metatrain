import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


def apply_recalibration_strategy(
    model: nn.Module,
    target: str,
    strategy: Dict[str, Any]
) -> nn.Module:
    """
    Apply the user-specified recalibration strategy to the LLPR-wrapped model.
    This function modifies the model in place based on the provided strategy.
    The strategy can be one of the following:
    - full: all model weights are retrained during calibration
    - tagged-only: only the model weights specified in a dictionary under this tag
       are retrained during calibration (useful for head-only calibration)
    - ens-only: only the ensemble linear layer weights are trained
    input model should be the LLPRUncertaintyModel object.
    strategy
    """

    method = strategy.get("strategy", "ens_only").lower()  

    # free-up last linear layers, freeze main model
    for name, module in model.llpr_ensemble_layers.items():
        for param in module.parameters():
            param.requires_grad = True
    for param in model.model.parameters():
        param.requires_grad = False            

    if method == "full":
        # Full finetuning, all parameters are trainable
        for param in model.model.parameters():
            param.requires_grad = True
    
    elif method == "tagged_only":
        tagged_param_list = strategy["tagged_only_weights"] 
        # only free up weights that contain string tags (useful for head-only)
        for name, param in model.model.named_parameters():
            if name in tagged_param_list:
                # above should be a list of valid parameter names
                param.requires_grad = True
                tagged_param_list.remove(name)
        if len(tagged_param_list) > 0:
            raise RuntimeError(
                f"Not all parameters have been matched within the wrapped model!\n"
                f"Remaining params: {tagged_param_list}"
            )

    elif method == "ens_only":
        # ll ensemble only
       pass

    return model
