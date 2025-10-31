from typing import NotRequired, Optional, Tuple

import torch
import torch.nn as nn
from typing_extensions import Literal, TypedDict


class LoRaFinetuneConfig(TypedDict):
    """Configuration for LoRA finetuning strategy."""

    rank: int
    """Rank of the LoRA matrices."""
    alpha: float
    """Scaling factor for the LoRA matrices."""
    target_modules: NotRequired[list[str]]


class HeadsFinetuneConfig(TypedDict):
    """Configuration for heads finetuning strategy."""

    head_modules: list[str]
    """List of module name prefixes for the prediction heads to finetune."""
    last_layer_modules: list[str]
    """List of module name prefixes for the last layers to finetune."""


class FullFinetuneHypers(TypedDict):
    """Hyperparameters to use full finetuning of PET models.

    This means all model parameters are trainable.
    """

    method: Literal["full"]
    """Finetuning method to use."""
    read_from: str
    """Path to the pretrained model checkpoint."""
    config: None
    """No configuration needed for full finetuning."""


class LoRaFinetuneHypers(TypedDict):
    """Hyperparameters for LoRA finetuning of PET models.

    Injects LoRA layers and finetunes only them.
    """

    method: Literal["lora"]
    """Finetuning method to use"""
    read_from: str
    """Path to the pretrained model checkpoint."""
    config: LoRaFinetuneConfig
    """Configuration for LoRA finetuning."""


class HeadsFinetuneHypers(TypedDict):
    """Hyperparameters for heads finetuning of PET models.

    Freezes all model parameters except for the prediction heads
    and last layers.
    """

    method: Literal["heads"]
    """Finetuning method to use."""
    read_from: str
    """Path to the pretrained model checkpoint."""
    config: HeadsFinetuneConfig
    """Configuration for heads finetuning."""


FinetuneHypers = FullFinetuneHypers | LoRaFinetuneHypers | HeadsFinetuneHypers


def apply_finetuning_strategy(model: nn.Module, strategy: FinetuneHypers) -> nn.Module:
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

    for param in model.parameters():
        param.requires_grad = True

    if strategy["method"] == "full":
        # Full finetuning, all parameters are trainable
        pass

    elif strategy["method"] == "lora":
        lora_config = strategy.get("config", {})
        lora_already_applied = any(isinstance(m, LoRALinear) for m in model.modules())
        if not lora_already_applied:
            model_device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
            model = inject_lora_layers(
                model,
                target_modules=tuple(
                    lora_config.get("target_modules", ["input_linear", "output_linear"])
                ),
                rank=lora_config.get("rank", 4),
                alpha=lora_config.get("alpha", 8),
                device=model_device,
                dtype=model_dtype,
            )

        # Freeze all except LoRA
        for name, param in model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

    elif strategy["method"] == "heads":
        heads_config = strategy.get(
            "config",
            {
                "head_modules": ["node_heads", "edge_heads"],
                "last_layer_modules": ["node_last_layers", "edge_last_layers"],
            },
        )

        head_keywords = heads_config.get("head_modules", [])
        last_layer_keywords = heads_config.get("last_layer_modules", [])

        for name, param in model.named_parameters():
            if any(name.startswith(kw) for kw in head_keywords + last_layer_keywords):
                param.requires_grad = True
            else:
                param.requires_grad = False

    else:
        raise ValueError(
            f"Unknown finetuning strategy: {strategy['method']}. Available methods "
            "are: 'full', 'lora', 'heads'."
        )

    model.finetune_config = strategy

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
