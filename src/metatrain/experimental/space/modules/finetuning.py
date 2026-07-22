from typing import Any, Dict, List, cast

import torch.nn as nn

from metatrain.pet.modules.finetuning import FinetuneHypers
from metatrain.pet.modules.finetuning import (
    apply_finetuning_strategy as _apply_finetuning_strategy,
)


# SPACE's ``Linear`` (see ``modules/layers.py``) wraps an inner ``nn.Linear``
# named ``linear_layer``, and its heads and last layers hang off the
# ``FakeGradientModel``. The defaults in ``metatrain.pet.modules.finetuning``
# name PET's modules, which match nothing in SPACE, so SPACE's own names are
# filled in here.
LORA_TARGET_MODULES = ["linear_layer"]
HEAD_MODULES = [
    "fake_gradient_model.module.heads",
    "fake_gradient_model.module.last_layers",
]
LAST_LAYER_MODULES: List[str] = []


def apply_finetuning_strategy(
    model: nn.Module, strategy: FinetuneHypers, apply_inherit_heads: bool = True
) -> nn.Module:
    """
    Apply the specified finetuning strategy to a SPACE model.

    This fills in the SPACE-specific module names that the strategy leaves
    unspecified, then delegates to :func:`metatrain.pet.modules.finetuning.
    apply_finetuning_strategy`. The resolved strategy (rather than the one the
    user wrote) is what ends up stored on the model, so that reloading a
    finetuned checkpoint targets the same modules.

    :param model: The SPACE model to be finetuned.
    :param strategy: A dictionary specifying the finetuning strategy.
    :param apply_inherit_heads: Whether to process the ``inherit_heads`` weight
        copy, a one-time initialization step. See
        :func:`metatrain.pet.modules.finetuning.apply_finetuning_strategy`.
    :return: The modified model with the finetuning strategy applied.
    """
    resolved: Dict[str, Any] = {**strategy}
    config: Dict[str, Any] = {**(resolved.get("config") or {})}

    if resolved["method"] == "lora":
        config.setdefault("target_modules", LORA_TARGET_MODULES)
    elif resolved["method"] == "heads":
        config.setdefault("head_modules", HEAD_MODULES)
        config.setdefault("last_layer_modules", LAST_LAYER_MODULES)

    resolved["config"] = config
    return _apply_finetuning_strategy(
        model, cast(FinetuneHypers, resolved), apply_inherit_heads=apply_inherit_heads
    )
