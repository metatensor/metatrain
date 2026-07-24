from metatrain.utils.scaler.checkpoints import update_per_property_scales


###########################
# MODEL ###################
###########################


def model_update_v1_v2(checkpoint: dict) -> None:
    """
    Update a v1 checkpoint to v2.

    :param checkpoint: The checkpoint to update.
    """
    update_per_property_scales(checkpoint)


def model_update_v2_v3(checkpoint: dict) -> None:
    """
    Update a v2 checkpoint to v3.

    Adds ``finetune_config`` to the state dicts (it was not present in v2).

    :param checkpoint: The checkpoint to update.
    """
    for key in ["model_state_dict", "best_model_state_dict"]:
        if (state_dict := checkpoint.get(key)) is not None:
            if "finetune_config" not in state_dict:
                state_dict["finetune_config"] = {}


#############################
# TRAINER ###################
#############################


def trainer_update_v1_v2(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 1 to version 2.

    :param checkpoint: The checkpoint to update.
    """
    checkpoint["train_hypers"]["max_atoms_per_batch"] = None
    checkpoint["train_hypers"]["min_atoms_per_batch"] = 0


def trainer_update_v2_v3(checkpoint: dict) -> None:
    """
    Update a v2 trainer checkpoint to v3.

    Adds empty ``finetune`` hypers if not present.

    :param checkpoint: The checkpoint to update.
    """
    if "finetune" not in checkpoint["train_hypers"]:
        checkpoint["train_hypers"]["finetune"] = {
            "read_from": None,
            "method": "full",
            "config": {},
            "inherit_heads": {},
        }
