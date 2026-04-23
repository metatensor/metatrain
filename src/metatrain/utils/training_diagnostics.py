import math
from typing import Dict, Optional

import torch
from metatensor.torch import TensorBlock, TensorMap


def _format_block_key(block_key) -> str:
    if len(block_key.names) == 0:
        return "<default>"
    return ", ".join(
        f"{name}={int(value)}"
        for name, value in zip(block_key.names, block_key.values, strict=True)
    )


def _get_values(tensor_block: TensorBlock, gradient_name: Optional[str]) -> torch.Tensor:
    if gradient_name is None:
        return tensor_block.values
    return tensor_block.gradient(gradient_name).values


def describe_supported_nonfinite_entries(
    predictions: Dict[str, TensorMap],
    targets: Dict[str, TensorMap],
    extra_data: Optional[Dict[str, TensorMap]] = None,
) -> Optional[str]:
    for target_name, tensor_map_targ in targets.items():
        if target_name not in predictions:
            continue

        tensor_map_pred = predictions[target_name]
        tensor_map_mask = None
        if extra_data is not None:
            tensor_map_mask = extra_data.get(f"{target_name}_mask")

        for block_key in tensor_map_targ.keys:
            target_block = tensor_map_targ.block(block_key)
            prediction_block = tensor_map_pred.block(block_key)
            mask_block = (
                tensor_map_mask.block(block_key) if tensor_map_mask is not None else None
            )

            for gradient_name in [None, *target_block.gradients_list()]:
                target_values = _get_values(target_block, gradient_name)
                prediction_values = _get_values(prediction_block, gradient_name)
                selected_mask = (
                    _get_values(mask_block, gradient_name).bool()
                    if mask_block is not None
                    else torch.ones_like(target_values, dtype=torch.bool)
                )

                if torch.isinf(target_values[selected_mask]).any():
                    channel = (
                        "values"
                        if gradient_name is None
                        else f"gradient '{gradient_name}'"
                    )
                    return (
                        "Targets contain +/-Inf on supervised entries for "
                        f"target '{target_name}', block {_format_block_key(block_key)}, "
                        f"{channel}."
                    )

                supported_mask = selected_mask & torch.isfinite(target_values)
                if not supported_mask.any():
                    continue

                if not torch.isfinite(prediction_values[supported_mask]).all():
                    channel = (
                        "values"
                        if gradient_name is None
                        else f"gradient '{gradient_name}'"
                    )
                    return (
                        "Predictions contain non-finite values on supervised entries "
                        f"for target '{target_name}', block {_format_block_key(block_key)}, "
                        f"{channel}."
                    )

    return None


def assert_finite_loss(
    loss_value: torch.Tensor,
    *,
    phase: str,
    predictions: Dict[str, TensorMap],
    targets: Dict[str, TensorMap],
    extra_data: Optional[Dict[str, TensorMap]] = None,
) -> None:
    if torch.isfinite(loss_value).all():
        return

    details = describe_supported_nonfinite_entries(predictions, targets, extra_data)
    message = f"Non-finite {phase} loss: {loss_value.item()}."
    if details is not None:
        message += f" {details}"
    raise ValueError(message)


def assert_finite_metrics(metrics: Dict[str, float], *, phase: str) -> None:
    for key, value in metrics.items():
        if math.isfinite(float(value)):
            continue
        raise ValueError(f"Non-finite {phase} metric '{key}' = {value}.")
