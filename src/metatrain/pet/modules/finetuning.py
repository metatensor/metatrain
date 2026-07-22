# mypy: disable-error-code=misc
# We ignore misc errors in this file because TypedDict
# with default values is not allowed by mypy.
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from typing_extensions import Literal, NotRequired, TypedDict

from metatrain.utils.data.target_info import TargetInfo


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


class NoFinetuneHypers(TypedDict):
    """Hypers that indicate that no finetuning is to be applied."""

    read_from: None = None
    """No finetuning is indicated by setting this argument to None.

    The rest of finetuning hyperparameters are then ignored.
    """
    method: NotRequired[Any]
    config: NotRequired[Any]
    inherit_heads: NotRequired[Any]


class FullFinetuneHypers(TypedDict):
    """Hyperparameters to use full finetuning of PET models.

    This means all model parameters are trainable.
    """

    method: Literal["full"] = "full"
    """Finetuning method to use."""
    read_from: str
    """Path to the pretrained model checkpoint."""
    config: NotRequired[Any]
    """No configuration needed for full finetuning."""
    inherit_heads: dict[str, str] = {}
    """Mapping from new trainable targets (keys) to the existing targets
    in the model (values).
    This allows for copying weights from the corresponding
    source heads to the destination heads instead of random initialization."""


class LoRaFinetuneHypers(TypedDict):
    """Hyperparameters for LoRA finetuning of PET models.

    Injects LoRA layers and finetunes only them.
    """

    method: Literal["lora"] = "lora"
    """Finetuning method to use"""
    read_from: str
    """Path to the pretrained model checkpoint."""
    config: LoRaFinetuneConfig
    """Configuration for LoRA finetuning."""
    inherit_heads: dict[str, str] = {}
    """Mapping from new trainable targets (keys) to the existing targets
    in the model (values).
    This allows for copying weights from the corresponding
    source heads to the destination heads instead of random initialization."""


class HeadsFinetuneHypers(TypedDict):
    """Hyperparameters for heads finetuning of PET models.

    Freezes all model parameters except for the prediction heads
    and last layers.
    """

    method: Literal["heads"] = "heads"
    """Finetuning method to use."""
    read_from: str
    """Path to the pretrained model checkpoint."""
    config: HeadsFinetuneConfig
    """Configuration for heads finetuning."""
    inherit_heads: dict[str, str] = {}
    """Mapping from new trainable targets (keys) to the existing targets
    in the model (values).
    This allows for copying weights from the corresponding
    source heads to the destination heads instead of random initialization."""


FinetuneHypers = FullFinetuneHypers | LoRaFinetuneHypers | HeadsFinetuneHypers


def compute_stale_targets(
    old_targets: Dict[str, TargetInfo], new_targets: Dict[str, TargetInfo]
) -> List[str]:
    """Determine which targets are made stale by a fine-tuning run.

    A target is stale if it was already present in the model before this run
    (``old_targets``) but is not part of the dataset the current run trains on
    (``new_targets``). When fine-tuning changes the backbone (``full``/``lora``
    methods), such targets' heads are no longer meaningful, since they were fit
    against a feature space the backbone no longer produces.

    :param old_targets: Targets already present in the model before this run.
    :param new_targets: Targets that are part of the dataset for this run.
    :return: Names of the stale targets.
    """
    return [key for key in old_targets if key not in new_targets]


def copy_head_weights(
    model: nn.Module, source_head_name: str, dest_head_name: str
) -> None:
    """Copy trainable head/last-layer parameter values between two targets.

    Every parameter whose fully-qualified name contains ``source_head_name`` is
    copied into the correspondingly-named parameter for ``dest_head_name``
    (found by substituting the target name in the parameter's name). Both targets
    must already exist in the model, with matching parameter shapes (i.e. the same
    target layout) -- this only copies values, it does not create or resize
    anything.

    :param model: The model whose parameters to copy between.
    :param source_head_name: Name of the target to copy weights from.
    :param dest_head_name: Name of the target to copy weights into.
    """
    model_parameters = dict(model.named_parameters())
    if not any(f".{source_head_name}." in name for name in model_parameters):
        raise ValueError(
            f"Weight copy was requested, but the source target name "
            f"'{source_head_name}' was not found in the model. Please specify "
            "the correct source target name."
        )
    if not any(f".{dest_head_name}." in name for name in model_parameters):
        raise ValueError(
            f"Weight copy was requested, but the destination target name "
            f"'{dest_head_name}' was not found in the model. Please specify "
            "the correct destination target name."
        )
    for name, param in model_parameters.items():
        if f".{source_head_name}." in name:
            corresponding_dest_name = name.replace(source_head_name, dest_head_name)
            if corresponding_dest_name in model_parameters:
                model_parameters[corresponding_dest_name].data.copy_(param.data)
            else:
                raise ValueError(
                    f"Destination head '{dest_head_name}' not found in model."
                )


def _add_backend_prefix(model: nn.Module, module_names: list[str]) -> list[str]:
    """Prepend the ``backend.`` prefix to module names that do not already have it.

    Some architectures (e.g. PET) keep their pure-PyTorch layers on a ``backend``
    submodule, while others (e.g. FlashMD) define them directly on the model. The
    prefix is only added for models that actually have a ``backend`` submodule, so
    that the same finetuning strategy can be reused across both kinds of models.

    :param model: The model the finetuning strategy is being applied to.
    :param module_names: List of module name prefixes as specified by the user.
    :return: The module name prefixes, prefixed with ``backend.`` if applicable.
    """
    if not hasattr(model, "backend"):
        return module_names
    return [
        name if name.startswith("backend.") else f"backend.{name}"
        for name in module_names
    ]


def apply_finetuning_strategy(
    model: nn.Module, strategy: FinetuneHypers, apply_inherit_heads: bool = True
) -> nn.Module:
    """
    Apply the specified finetuning strategy to the model.
    This function modifies the model in place based on the provided strategy.

    :param model: The model to be finetuned.
    :param strategy: A dictionary specifying the finetuning strategy.
        The strategy method can be one of the following:
        - lora: Inject LoRA layers into the model, or reapply training if already
            present.
        - heads: Freeze all parameters except for the heads and last layers.
        - full: All parameters are trainable.
        Additionally, the strategy can include an "inherit_heads" key,
        which is a dictionary mapping the new trainable targets to the existing
        targets in the model. This allows for copying weights from the corresponding
        source heads to the destination heads instead of random initialization.
    :param apply_inherit_heads: Whether to process the "inherit_heads" weight copy
        (and the removal of targets left stale by a backbone-altering finetuning
        run, see :func:`compute_stale_targets`). Both are one-time initialization
        steps that must run exactly once, when finetuning actually starts. Callers
        that re-apply an already-active strategy after reloading a checkpoint
        (e.g. to restore the trainable/frozen parameter state) should pass
        ``False``: by then, any inherited-from source target may already have been
        pruned, so redoing the copy would fail (or silently clobber trained
        weights if the source were still around).
    :return: The modified model with the finetuning strategy applied.
    """

    for param in model.parameters():
        param.requires_grad = True

    if strategy["method"] == "full":
        # Full finetuning, all parameters are trainable
        pass

    elif strategy["method"] == "lora":
        lora_config = strategy["config"]
        target_modules = tuple(
            lora_config.get("target_modules", ["input_linear", "output_linear"])
        )
        lora_already_applied = any(isinstance(m, LoRALinear) for m in model.modules())
        if not lora_already_applied:
            model_device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
            model = inject_lora_layers(
                model,
                target_modules=target_modules,
                rank=lora_config.get("rank", 4),
                alpha=lora_config.get("alpha", 8),
                device=model_device,
                dtype=model_dtype,
            )
            if not any(isinstance(m, LoRALinear) for m in model.modules()):
                raise ValueError(
                    "No LoRA layers were injected: no modules matching "
                    f"'target_modules' {list(target_modules)} were found in the "
                    "model. Please check that these module names are correct."
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

        # On architectures like PET, the heads and last layers actually live on
        # the pure-PyTorch ``backend`` submodule, but users should only need to
        # refer to them by their old, un-prefixed names, so the ``backend.``
        # prefix is added automatically here when applicable.
        head_keywords = _add_backend_prefix(model, heads_config.get("head_modules", []))
        last_layer_keywords = _add_backend_prefix(
            model, heads_config.get("last_layer_modules", [])
        )
        all_keywords = head_keywords + last_layer_keywords

        matched_any = False
        for name, param in model.named_parameters():
            if any(name.startswith(kw) for kw in all_keywords):
                param.requires_grad = True
                matched_any = True
            else:
                param.requires_grad = False

        if not matched_any:
            raise ValueError(
                "No parameters were found matching the specified 'head_modules' "
                f"({heads_config.get('head_modules', [])}) or "
                f"'last_layer_modules' ({heads_config.get('last_layer_modules', [])}). "
                "Please check that these module name prefixes are correct."
            )

    else:
        raise ValueError(
            f"Unknown finetuning strategy: {strategy['method']}. Available methods "
            "are: 'full', 'lora', 'heads'."
        )

    model.finetune_config = strategy

    inherit_heads_config = strategy["inherit_heads"]
    if apply_inherit_heads and inherit_heads_config:
        for dest_head_name, source_head_name in inherit_heads_config.items():
            copy_head_weights(model, source_head_name, dest_head_name)

    # Targets not part of this run's dataset are dropped now that weight
    # inheritance (if any) has had a chance to copy from their heads: with
    # ``full``/``lora``, the backbone has changed, so these targets' heads are no
    # longer meaningful. ``restart`` always stashes the list on the model rather
    # than removing right away, since it runs before this function (and before
    # ``inherit_heads`` above needs the stale heads to still be present); with
    # ``heads`` the backbone is unchanged, so stale targets are left alone here.
    stale_targets = getattr(model, "_stale_finetune_targets", None)
    if stale_targets and strategy["method"] in ("full", "lora"):
        for target_name in stale_targets:
            model.remove_output(target_name)
            if target_name in model.target_names:
                model.target_names.remove(target_name)
            model.dataset_info.targets.pop(target_name, None)
            for additive_model in model.additive_models:
                if target_name in additive_model.outputs:
                    additive_model.remove_output(target_name)
            if target_name in model.scaler.outputs:
                model.scaler.remove_output(target_name)
        model._stale_finetune_targets = []

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
