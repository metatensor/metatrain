import metatensor.torch as mts


###########################
# MODEL ###################
###########################


def model_update_v1_v2(checkpoint: dict) -> None:
    """
    Update model checkpoint from version 1 to version 2.

    The only change is to fix a typo in the unit of momentum targets.

    :param checkpoint: The checkpoint to update.
    """
    for _, target in checkpoint["model_data"]["dataset_info"].targets.items():
        if target.unit == "(eV*u)^1/2":
            target.unit = "(eV*u)^(1/2)"


def model_update_v2_v3(checkpoint: dict) -> None:
    """
    Update model checkpoint from version 2 to version 3.

    :param checkpoint: The checkpoint to update.
    """
    # Adding the attention_temperature hyperparameter if not present
    if "attention_temperature" not in checkpoint["model_data"]["model_hypers"]:
        checkpoint["model_data"]["model_hypers"]["attention_temperature"] = 1.0


def model_update_v3_v4(checkpoint: dict) -> None:
    """
    Update a v3 checkpoint to v4.

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
    # remove all entries in the loss `sliding_factor`
    old_loss_hypers = checkpoint["train_hypers"]["loss"].copy()
    dataset_info = checkpoint["model_data"]["dataset_info"]
    new_loss_hypers = {}

    for target_name in dataset_info.targets.keys():
        # retain everything except sliding_factor for each target
        new_loss_hypers[target_name] = {
            k: v
            for k, v in old_loss_hypers[target_name].items()
            if k != "sliding_factor"
        }

    checkpoint["train_hypers"]["loss"] = new_loss_hypers


def trainer_update_v2_v3(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 2 to version 3.

    :param checkpoint: The checkpoint to update.
    """
    # - Rename ``fixed_composition_weights`` to ``atomic_baseline``.
    atomic_baseline = checkpoint["train_hypers"].pop("fixed_composition_weights")
    checkpoint["train_hypers"]["atomic_baseline"] = atomic_baseline


def trainer_update_v3_v4(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 3 to version 4.

    :param checkpoint: The checkpoint to update.
    """
    checkpoint["train_hypers"]["batch_atom_bounds"] = [None, None]


def trainer_update_v4_v5(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 4 to version 5.

    :param checkpoint: The checkpoint to update.
    """
    # Adding the empty finetune config if not present
    if "finetune" not in checkpoint["train_hypers"]:
        checkpoint["train_hypers"]["finetune"] = {
            "read_from": None,
            "method": "full",
            "config": {},
            "inherit_heads": {},
        }
