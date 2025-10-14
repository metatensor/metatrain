from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


def apply_finetuning_strategy(model: nn.Module, strategy: Dict[str, Any]) -> nn.Module:
    """
    Apply the specified finetuning strategy to the model.
    This function modifies the model in place based on the provided strategy.

    :param model: The model to be finetuned.
    :param strategy: A dictionary specifying the finetuning strategy.
        The strategy can be one of the following:
        - lora: Inject LoRA layers into the model, or reapply training if already
            present.
        - heads: Freeze all parameters except for the heads and last layers.
    :return: The modified model with the finetuning strategy applied.
    """
    method = strategy.get("method", "full").lower()

    for param in model.parameters():
        param.requires_grad = True

    if method == "full":
        # Full finetuning, all parameters are trainable
        pass

    elif method == "lora":
        strategy_cfg = strategy.get("config", {})
        lora_already_applied = any(isinstance(m, LoRALinear) for m in model.modules())
        if not lora_already_applied:
            model_device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
            model = inject_lora_layers(
                model,
                target_modules=tuple(
                    strategy_cfg.get(
                        "target_modules", ("input_linear", "output_linear")
                    )
                ),
                rank=strategy_cfg.get("rank", 4),
                alpha=strategy_cfg.get("alpha", 8),
                device=model_device,
                dtype=model_dtype,
            )

        # Freeze all except LoRA
        for name, param in model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

    elif method == "heads":
        strategy_cfg = strategy.get(
            "config",
            {
                "head_modules": ["node_heads", "edge_heads"],
                "last_layer_modules": ["node_last_layers", "edge_last_layers"],
            },
        )

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
            "are: 'full', 'lora', 'heads'."
        )

    model.finetune_config = strategy

    inherit_heads_config = strategy.get("inherit_heads", {})
    if inherit_heads_config:
        for dest_target_name, source_target_name in inherit_heads_config.items():
            model_parameters = dict(model.named_parameters())
            for name, param in model_parameters.items():
                if f".{source_target_name}." in name:
                    corresponding_dest_name = name.replace(
                        source_target_name, dest_target_name
                    )
                    if corresponding_dest_name in model_parameters:
                        model_parameters[corresponding_dest_name].data.copy_(param.data)
                    else:
                        raise ValueError(
                            f"Destination head '{dest_target_name}' not found in model."
                        )
    return model


def inject_lora_layers(
    model: nn.Module,
    target_modules: Tuple[str, ...] = ("input_linear", "output_linear"),
    rank: int = 4,
    alpha: float = 1.0,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """
    Inject LoRA layers into the model.
    This function replaces the specified linear layers in the model with
    LoRALinear layers.

    :param model: The model to modify.
    :param target_modules: A tuple of strings specifying the names of the attributes of
        the modules to be replaced with LoRA layers.
    :param rank: The rank of the LoRA matrices.
    :param alpha: The scaling factor for the LoRA matrices.
    :param device: The device to which the LoRA layers should be moved. If None, the
        LoRA layers will be on the same device as the original model.
    :param dtype: The data type to which the LoRA layers should be converted. If
        None, the LoRA layers will have the same dtype as the original model.
    :return: The modified model with LoRA layers injected.
    """
    for _, module in model.named_modules():
        for attr in target_modules:
            if hasattr(module, attr):
                linear = getattr(module, attr)
                if isinstance(linear, nn.Linear):
                    lora_linear = LoRALinear(linear, rank=rank, alpha=alpha)
                    lora_linear = lora_linear.to(dtype=dtype, device=device)
                    setattr(module, attr, lora_linear)
    return model


class LoRALinear(nn.Module):
    """
    LoRA Linear layer.
    This is a wrapper around nn.Linear that adds LoRA functionality.
    LoRA is a technique for low-rank adaptation of large language models.
    It allows for efficient fine-tuning of large models by injecting low-rank
    matrices into the model's weights.

    :param linear_layer: The original linear layer to be wrapped.
    :param rank: The rank of the LoRA matrices.
    :param alpha: The scaling factor for the LoRA matrices.
    """

    def __init__(self, linear_layer: nn.Module, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.linear = linear_layer
        self.lora_A = nn.Linear(linear_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, linear_layer.out_features, bias=False)
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.scaling * self.lora_B(self.lora_A(x))
