from metatrain.scaler.checkpoints import update_per_property_scales


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

    :param checkpoint: The checkpoint to update.
    """
    # Remove the "dpa3_model" key from the hypers dictionary if it exists
    if "dpa3_model" not in checkpoint["model_data"]["model_hypers"]:
        checkpoint["model_data"]["model_hypers"]["dpa3_model"] = None
    if "dpa3_model_branch" not in checkpoint["model_data"]["model_hypers"]:
        checkpoint["model_data"]["model_hypers"]["dpa3_model_branch"] = None


###########################
# TRAINER #################
###########################


def trainer_update_v1_v2(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 1 to version 2.

    :param checkpoint: The checkpoint to update.
    """
    checkpoint["train_hypers"]["max_atoms_per_batch"] = None
    checkpoint["train_hypers"]["min_atoms_per_batch"] = 0
