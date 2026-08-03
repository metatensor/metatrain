"""
Utility functions to update checkpoints of architectures
when it is the scaler's fault.
"""

import metatensor.torch as mts
import torch


def model_update_v1_v2(checkpoint: dict, prefix: str = "") -> None:
    """
    Update model checkpoint from version 1 to version 2.

    The model now uses metatensor's ``nn.Module`` and ``register_buffer`` instead of
    manually tracking the data.

    :param checkpoint: The checkpoint to update.
    :param prefix: Prefix prepended to the state_dict keys of the scaler (e.g.
        ``"scaler."`` when the scaler is nested inside another architecture's
        ``self.scaler``).
    """
    scaler_key = f"{prefix}model"

    for key in ["model_state_dict", "best_model_state_dict"]:
        if (state_dict := checkpoint.get(key)) is None:
            continue

        # If both model_state_dict and best_model_state_dict point to the same
        # dict, the upgrade was already applied in the first iteration.
        if f"{scaler_key}._mts_helper" in state_dict:
            continue

        dummy_buffer = state_dict[f"{prefix}dummy_buffer"]
        empty_tensor = torch.zeros(
            0, dtype=dummy_buffer.dtype, device=dummy_buffer.device
        )

        extra_state: dict[str, dict] = {
            "scales": {},
            "per_target_scales": {},
            "per_property_scales": {},
        }

        for target_name in checkpoint["model_data"]["dataset_info"].targets:
            scales_key = f"{prefix}{target_name}_scaler_buffer"
            per_target_key = f"{prefix}{target_name}_per_target_scaler_buffer"
            per_property_key = f"{prefix}{target_name}_per_property_scaler_buffer"

            if scales_key not in state_dict:
                continue

            extra_state["scales"][target_name] = (
                "metatensor.TensorMap",
                state_dict.pop(scales_key),
                empty_tensor,
            )
            extra_state["per_target_scales"][target_name] = (
                "metatensor.TensorMap",
                state_dict.pop(per_target_key),
                empty_tensor,
            )
            extra_state["per_property_scales"][target_name] = (
                "metatensor.TensorMap",
                state_dict.pop(per_property_key),
                empty_tensor,
            )

        state_dict[f"{scaler_key}._mts_helper"] = empty_tensor
        state_dict[f"{scaler_key}._extra_state"] = extra_state


def update_per_property_scales(checkpoint: dict, scaler_key: str = "scaler") -> None:
    """
    Updates architecture checkpoints to add per-property scales,
    so that they comply with the scaler changes introduced in
    https://github.com/metatensor/metatrain/pull/1107.

    :param checkpoint: The architecture checkpoint to update.
    :param scaler_key: The key under which the scaler is stored in the state_dict
      of the model.
    """
    targets_scaled = checkpoint.get("train_hypers", {}).get("scale_targets", True)
    if targets_scaled:
        # If the model checkpoint can output targets with multiple blocks and/or
        # multiple properties per-block, this version of metatrain cannot be used.
        # This doesn't affect MLIP checkpoints.
        for target_name, target_info in checkpoint["model_data"][
            "dataset_info"
        ].targets.items():
            layout = target_info.layout
            if len(layout.keys) > 1 or len(layout[0].properties) > 1:
                raise ValueError(
                    f"Target '{target_name}' has multiple blocks or multiple "
                    "properties per block. Upgrading checkpoints for such targets is "
                    "not supported, as it would require re-computing per-target "
                    "scales from the original training data. Please install from "
                    "source the older version of metatrain (before the "
                    "per-target/per-property scale separation)."
                )

    # For single-block, single-property targets (e.g. MLIPs): the old `scales`
    # TensorMap can be used directly as `per_target_scales`, and `per_property_scales`
    # is set to 1 (since there is only one property, per-property scales are trivially
    # 1 by definition).
    for key in ["model_state_dict", "best_model_state_dict"]:
        if (state_dict := checkpoint.get(key)) is not None:
            for target_name in checkpoint["model_data"]["dataset_info"].targets:
                buffer_key = f"{scaler_key}.{target_name}_scaler_buffer"
                if buffer_key not in state_dict:
                    continue
                scales_tm = mts.load_buffer(state_dict[buffer_key])
                per_target_tm = scales_tm.copy()
                per_property_tm = mts.ones_like(scales_tm)
                state_dict[f"{scaler_key}.{target_name}_per_target_scaler_buffer"] = (
                    mts.save_buffer(mts.make_contiguous(per_target_tm))
                )
                state_dict[f"{scaler_key}.{target_name}_per_property_scaler_buffer"] = (
                    mts.save_buffer(mts.make_contiguous(per_property_tm))
                )
