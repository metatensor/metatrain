from typing import Any, Dict

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

    else:
        raise ValueError(
            f"Unknown finetuning strategy: {method}. Available methods are: 'full'."
        )

    model.finetune_config = strategy

    return model
