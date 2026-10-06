"""Checkpoint fixes for target names that metatomic has since renamed."""

from typing import Any

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatrain.utils.data.target_info import TargetInfo


# ``non_conservative_forces`` was renamed to ``non_conservative_force`` in
# metatomic 0.1.12 / metatrain #1153. Checkpoints trained before that still
# store the plural name as a property label (and sometimes as the target key).
LEGACY_NON_CONSERVATIVE_FORCE = "non_conservative_forces"
NON_CONSERVATIVE_FORCE = "non_conservative_force"


def rename_label_column(
    tensor_map: TensorMap, old_name: str, new_name: str
) -> TensorMap:
    """Return ``tensor_map`` with every ``old_name`` label column renamed.

    The input is returned unchanged when ``old_name`` does not appear.

    :param tensor_map: Tensor map whose label columns may use ``old_name``.
    :param old_name: Label name to replace.
    :param new_name: Label name to use instead.
    :return: ``tensor_map`` itself when nothing changed, otherwise a copy.
    """

    def column(labels: Labels) -> Labels:
        if old_name not in labels.names:
            return labels
        names = [new_name if name == old_name else name for name in labels.names]
        return Labels(names, labels.values)

    def block_uses_name(block: TensorBlock) -> bool:
        if old_name in block.samples.names or old_name in block.properties.names:
            return True
        if any(old_name in component.names for component in block.components):
            return True
        return any(block_uses_name(gradient) for _, gradient in block.gradients())

    if old_name not in tensor_map.keys.names and not any(
        block_uses_name(block) for block in tensor_map.blocks()
    ):
        return tensor_map

    blocks = []
    for block in tensor_map.blocks():
        renamed = TensorBlock(
            values=block.values,
            samples=column(block.samples),
            components=[column(component) for component in block.components],
            properties=column(block.properties),
        )
        for gradient_name, gradient in block.gradients():
            renamed.add_gradient(
                gradient_name,
                TensorBlock(
                    values=gradient.values,
                    samples=column(gradient.samples),
                    components=[column(component) for component in gradient.components],
                    properties=column(gradient.properties),
                ),
            )
        blocks.append(renamed)
    return TensorMap(keys=column(tensor_map.keys), blocks=blocks)


def _rename_target_info(target_info: TargetInfo) -> TargetInfo:
    layout = rename_label_column(
        target_info.layout, LEGACY_NON_CONSERVATIVE_FORCE, NON_CONSERVATIVE_FORCE
    )
    if layout is target_info.layout:
        return target_info
    return TargetInfo(
        layout=layout,
        quantity=target_info.quantity,
        unit=target_info.unit,
        description=target_info.description,
    )


def _rename_mapping(mapping: Any) -> None:
    """Rename the legacy force key inside dicts and lists stored in a checkpoint.

    :param mapping: A checkpoint fragment. Nested dicts and lists are updated
        in place. Other objects are ignored.
    """
    if isinstance(mapping, dict):
        if (
            LEGACY_NON_CONSERVATIVE_FORCE in mapping
            and NON_CONSERVATIVE_FORCE not in mapping
        ):
            mapping[NON_CONSERVATIVE_FORCE] = mapping.pop(LEGACY_NON_CONSERVATIVE_FORCE)
        for value in mapping.values():
            _rename_mapping(value)
    elif isinstance(mapping, list):
        for index, value in enumerate(mapping):
            if value == LEGACY_NON_CONSERVATIVE_FORCE:
                mapping[index] = NON_CONSERVATIVE_FORCE
            else:
                _rename_mapping(value)


def _rename_buffer(value: torch.Tensor) -> torch.Tensor:
    try:
        tensor_map = mts.load_buffer(value.detach().to("cpu"))
    except Exception:
        return value
    renamed = rename_label_column(
        tensor_map, LEGACY_NON_CONSERVATIVE_FORCE, NON_CONSERVATIVE_FORCE
    )
    if renamed is tensor_map:
        return value
    saved = mts.save_buffer(mts.make_contiguous(renamed))
    return saved.to(device=value.device)


def _rename_state_dict(state_dict: dict) -> dict:
    updated = {}
    for key, value in state_dict.items():
        new_key = key.replace(LEGACY_NON_CONSERVATIVE_FORCE, NON_CONSERVATIVE_FORCE)
        if (
            isinstance(value, torch.Tensor)
            and key.endswith("_buffer")
            and not key.endswith("dummy_buffer")
        ):
            value = _rename_buffer(value)
        elif not isinstance(value, torch.Tensor):
            _rename_mapping(value)
        updated[new_key] = value
    return updated


def upgrade_legacy_non_conservative_force(checkpoint: dict) -> None:
    """Rename ``non_conservative_forces`` to ``non_conservative_force`` in-place.

    Covers dataset layouts, serialized scaler and composition buffers, state-dict
    key paths, and target-keyed training hypers. Nested ``wrapped_model_checkpoint``
    entries (LLPR) are updated as well. Checkpoints that already use the singular
    name are left unchanged.

    :param checkpoint: Checkpoint dictionary to update.
    """
    model_data = checkpoint.get("model_data")
    if isinstance(model_data, dict):
        dataset_info = model_data.get("dataset_info")
        if dataset_info is not None and hasattr(dataset_info, "targets"):
            targets = dataset_info.targets
            if (
                LEGACY_NON_CONSERVATIVE_FORCE in targets
                and NON_CONSERVATIVE_FORCE not in targets
            ):
                targets[NON_CONSERVATIVE_FORCE] = targets.pop(
                    LEGACY_NON_CONSERVATIVE_FORCE
                )
            for name, target_info in list(targets.items()):
                targets[name] = _rename_target_info(target_info)

    for key in ("model_state_dict", "best_model_state_dict"):
        state_dict = checkpoint.get(key)
        if isinstance(state_dict, dict):
            checkpoint[key] = _rename_state_dict(state_dict)

    _rename_mapping(checkpoint.get("train_hypers"))

    wrapped = checkpoint.get("wrapped_model_checkpoint")
    if isinstance(wrapped, dict):
        upgrade_legacy_non_conservative_force(wrapped)
