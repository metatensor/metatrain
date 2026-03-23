###########################
# MODEL ###################
###########################


def model_update_v1_v2(checkpoint: dict) -> None:
    """
    Update a v1 PhACE model checkpoint to v2.

    Adds ``finetune_config`` to state dicts (was not present in v1).

    :param checkpoint: The checkpoint to update.
    """
    for key in ["model_state_dict", "best_model_state_dict"]:
        if (state_dict := checkpoint.get(key)) is not None:
            if "finetune_config" not in state_dict:
                state_dict["finetune_config"] = {}


###########################
# TRAINER #################
###########################


def trainer_update_v1_v2(checkpoint: dict) -> None:
    """
    Update a v1 PhACE trainer checkpoint to v2.

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
