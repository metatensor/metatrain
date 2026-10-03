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
    if "dpa3_model_branch" not in checkpoint["model_data"]["model_hypers"]:
        checkpoint["model_data"]["model_hypers"]["dpa3_model_branch"] = None

def model_update_v3_v4(checkpoint: dict) -> None:
    """
    Update a v3 checkpoint to v4.

    It removes the additive models and scaler from the model checkpoint,
    as this is now handled by the MetatrainModel wrapper.

    :param checkpoint: The checkpoint to update.
    """
    removed_prefixes = (
        "additive_models.",
        "scaler.",
    )
    for key in ["model_state_dict", "best_model_state_dict"]:
        if (state_dict := checkpoint.get(key)) is not None:
            for k in list(state_dict):
                for prefix in removed_prefixes:
                    if k.startswith(prefix):
                        state_dict.pop(k)

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


def trainer_update_v2_v3(checkpoint: dict) -> None:
    """
    Deprecate the ``distributed`` hyperparameter.

    So that it doesn't show up in the restarting options.

    :param checkpoint: The checkpoint to update.
    """
    train_hypers = checkpoint["train_hypers"]
    if "distributed" in train_hypers and train_hypers["distributed"] is None:
        train_hypers.pop("distributed")
