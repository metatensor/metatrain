import logging
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


def apply_finetuning_strategy(model: nn.Module, strategy: Dict[str, Any]) -> nn.Module:
    """
    Apply the specified finetuning strategy to the model.
    This function modifies the model in place based on the provided strategy.
    The strategy can be one of the following:
    - lora: Inject LoRA layers into the model, or reapply training if already present.
    - heads: Freeze all parameters except for the heads and last layers.
    """
    method = strategy["method"].lower()

    for param in model.parameters():
        param.requires_grad = True

    if method == "lora":
        strategy_cfg = strategy.get("config", {})
        lora_already_applied = any(isinstance(m, LoRALinear) for m in model.modules())
        if not lora_already_applied:
            model = inject_lora_layers(
                model,
                target_modules=tuple(
                    strategy_cfg.get(
                        "target_modules", ("input_linear", "output_linear")
                    )
                ),
                rank=strategy_cfg.get("rank", 4),
                alpha=strategy_cfg.get("alpha", 8),
            )

        # Freeze all except LoRA
        for name, param in model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

    elif method == "heads":
        strategy_cfg = strategy.get("config", {})

        head_keywords = strategy_cfg.get("head_modules", [])
        last_layer_keywords = strategy_cfg.get("last_layer_modules", [])

        for name, param in model.named_parameters():
            if any(name.startswith(kw) for kw in head_keywords + last_layer_keywords):
                param.requires_grad = True
            else:
                param.requires_grad = False

    else:
        raise ValueError(
            f"Unknown finetuning strategy: {method}. Available methods "
            "are: lora, heads."
        )
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Applied finetuning strategy: {method}")
    logger.info(
        f"Number of trainable parameters: {num_trainable_params} "
        f"[{num_trainable_params / num_params:.2%} %]"
    )

    return model


def inject_lora_layers(
    model: nn.Module,
    target_modules: Tuple[str, ...] = ("input_linear", "output_linear"),
    rank: int = 4,
    alpha: float = 1.0,
) -> nn.Module:
    """
    Inject LoRA layers into the model.
    This function replaces the specified linear layers in the model with
    LoRALinear layers.
    """
    for _, module in model.named_modules():
        for attr in target_modules:
            if hasattr(module, attr):
                linear = getattr(module, attr)
                if isinstance(linear, nn.Linear):
                    setattr(module, attr, LoRALinear(linear, rank=rank, alpha=alpha))
    return model


class LoRALinear(nn.Module):
    """
    LoRA Linear layer.
    This is a wrapper around nn.Linear that adds LoRA functionality.
    LoRA is a technique for low-rank adaptation of large language models.
    It allows for efficient fine-tuning of large models by injecting low-rank
    matrices into the model's weights.
    """

    def __init__(self, linear_layer: nn.Module, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.linear = linear_layer
        self.lora_A = nn.Linear(linear_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, linear_layer.out_features, bias=False)
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.scaling * self.lora_B(self.lora_A(x))
