import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


def apply_recalibration_strategy(model: nn.Module, strategy: Dict[str, Any]) -> nn.Module:
    """
    Apply the user-specified recalibration strategy to the LLPR-wrapped model.
    This function modifies the model in place based on the provided strategy.
    The strategy can be one of the following:
    - full: 
    - tagged-only:
    - ens-only:
    """
    method = strategy.get("method", "full").lower()

    for param in model.parameters():
        param.requires_grad = True

    if method == "full":
        # Full finetuning, all parameters are trainable
        pass

    elif method == "tagged-only":
        # only free up weights that contain string tags (useful for head-only)
        pass

    elif method == "ens-only":
        # ll ensemble only
        pass


