"""
Utility functions to update checkpoints of architectures
when it is the scaler's fault.
"""

import metatensor.torch as mts


def update_per_property_scales(checkpoint: dict, scaler_key: str = "scaler") -> None:
    """
    Updates architecture checkpoints to add per-property scales,
    so that they comply with the scaler changes introduced in
    https://github.com/metatensor/metatrain/pull/1107.

    :param checkpoint: The architecture checkpoint to update.
    :param scaler_key: The key under which the scaler is stored in the state_dict
      of the model.
    """
    if checkpoint["train_hypers"]["scale_targets"]:
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
