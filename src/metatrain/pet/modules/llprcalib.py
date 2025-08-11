import logging
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

def apply_llpr_calib_strategy(model: nn.Module, strategy: Dict[str, Any]) -> nn.Module:
    """
    Modify LLPR-wrapped model to directly incorporate LLPR ensemble weights in the architecture.
    """
    strategy = strategy["method"]

    llpr_ens_already_applied = any(isinstance(m, LLPREnsembleLinear) for m in model.modules())

    if not llpr_ens_already_applied:
        model = inject_llpr_ensemble(model)

    for param in model.parameters():
        param.requires_grad = True

    if strategy == "weights_only":
        for name, param in model.named_parameters():
            if "llpr_" not in name:
                param.requires_grad = False

    elif strategy == "heads":
        for name, param in model.named_parameters():
            if "heads" not in name or "llpr_" not in name:
                param.requires_grad = False

    else:
        raise ValueError(
            f"Unknown llpr calibration strategy: {strategy}. Available methods "
            "are: weights_only, heads."
        )

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Entering LLPR ensemble calibration.")
    logger.info(
        f"Number of trainable parameters: {num_trainable_params} "
        f"[{num_trainable_params / num_params:.2%} %]"
    )

    return model


def inject_llpr_ensemble(
    model: nn.Module,
) -> nn.Module:
    """
    Inject LLPR ensemble to model architecture.
    """
    for _, module in model.named_modules():
        for attr in target_modules:
            if hasattr(module, attr):
                linear = getattr(module, attr)
                if isinstance(linear, nn.Linear):
                    setattr(module, attr, LoRALinear(linear, rank=rank, alpha=alpha))
    return model


class LLPREnsembleLinear(nn.Module):
    """
    LLPR ensemble Linear layer.
    This is a wrapper around nn.Linear that adds LoRA functionality.
    LoRA is a technique for low-rank adaptation of large language models.
    It allows for efficient fine-tuning of large models by injecting low-rank
    matrices into the model's weights.
    """

    def __init__(self, last_layer: nn.Module, ens_weights: torch.Tensor):
        super().__init__()
        self.linear = last_layer
        self.llpr_ensemble = []
        for ii in len(ens_weights):
            self.llpr_ensemble.append(
                nn.Linear(last_layer.in_features, ens_weights.shape[1], bias=False)
            )
        with torch.no_grad():
            for ii, member in enumerate(self.llpr_ensemble):
                member.weight = ens_weights[ii]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for member in self.llpr_ensemble:
            outputs.append(member(x))
        return torch.vstack(outputs)
