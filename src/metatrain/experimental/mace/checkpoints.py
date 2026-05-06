import metatensor.torch as mts


###########################
# MODEL ###################
###########################


def model_update_v1_v2(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 1 to version 2.

    :param checkpoint: The checkpoint to update.
    """
    # In v1 there was a huge bug where `edge_irreps` was passed
    # to MACE even if it was None. Here we keep this functionality
    # to update from v1 to v2, although people should really just
    # retrain, because the v1 situation was very bad.
    if checkpoint["model_data"]["hypers"]["edge_irreps"] is None:
        checkpoint["model_data"]["hypers"]["edge_irreps"] = ""


def model_update_v2_v3(checkpoint: dict) -> None:
    """
    Update a v2 checkpoint to v3.

    :param checkpoint: The checkpoint to update.
    """
    # If the model checkpoint can output targets with multiple blocks and/or multiple
    # properties per-block, this version of metatrain cannot be used. This doesn't
    # affect MLIP checkpoints.
    for target_name, target_info in checkpoint["model_data"][
        "dataset_info"
    ].targets.items():
        layout = target_info.layout
        if len(layout.keys) > 1 or len(layout[0].properties) > 1:
            raise ValueError(
                f"Target '{target_name}' has multiple blocks or multiple properties "
                f"per block. Upgrading checkpoints for such targets is not supported, "
                f"as it would require re-computing per-target scales from the original "
                f"training data. Please install from source the older version of "
                f"metatrain (before the per-target/per-property scale separation)."
            )

    # For single-block, single-property targets (e.g. MLIPs): the old `scales`
    # TensorMap can be used directly as `per_target_scales`, and `per_property_scales`
    # is set to 1 (since there is only one property, per-property scales are trivially
    # 1 by definition).
    for key in ["model_state_dict", "best_model_state_dict"]:
        if (state_dict := checkpoint.get(key)) is not None:
            for target_name in checkpoint["model_data"]["dataset_info"].targets:
                buffer_key = f"scaler.{target_name}_scaler_buffer"
                if buffer_key not in state_dict:
                    continue
                scales_tm = mts.load_buffer(state_dict[buffer_key])
                per_target_tm = scales_tm.copy()
                per_property_tm = mts.ones_like(scales_tm)
                state_dict[f"scaler.{target_name}_per_target_scaler_buffer"] = (
                    mts.save_buffer(mts.make_contiguous(per_target_tm))
                )
                state_dict[f"scaler.{target_name}_per_property_scaler_buffer"] = (
                    mts.save_buffer(mts.make_contiguous(per_property_tm))
                )


###########################
# TRAINER #################
###########################


def trainer_update_v1_v2(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 1 to version 2.

    :param checkpoint: The checkpoint to update.
    """
    checkpoint["train_hypers"]["batch_atom_bounds"] = [None, None]
