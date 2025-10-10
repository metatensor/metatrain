###########################
# MODEL ###################
###########################


def model_update_v1_v2(checkpoint: dict) -> None:
    """
    Update a v1 checkpoint to v2.

    :param checkpoint: The checkpoint to update.
    """
    for key in ["model_state_dict", "best_model_state_dict"]:
        if (state_dict := checkpoint.get(key)) is not None:
            if "additive_models.0.model.type_to_index" not in state_dict:
                state_dict["additive_models.0.model.type_to_index"] = state_dict.pop(
                    "additive_models.0.type_to_index"
                )


def model_update_v2_v3(checkpoint: dict) -> None:
    """
    Update a v2 checkpoint to v3.

    :param checkpoint: The checkpoint to update.
    """
    for key in ["model_state_dict", "best_model_state_dict"]:
        if (state_dict := checkpoint.get(key)) is not None:
            if "train_hypers" in state_dict:
                finetune_config = state_dict["train_hypers"].get("finetune", {})
            else:
                finetune_config = {}
            state_dict["finetune_config"] = finetune_config


def model_update_v3_v4(checkpoint: dict) -> None:
    """
    Update a v3 checkpoint to v4.

    :param checkpoint: The checkpoint to update.
    """
    checkpoint["epoch"] = checkpoint.get("epoch")
    checkpoint["best_epoch"] = checkpoint.get("best_epoch")

    if checkpoint["best_model_state_dict"] is not None:
        checkpoint["best_model_state_dict"] = checkpoint.get("best_model_state_dict")


def model_update_v4_v5(checkpoint: dict) -> None:
    """
    Update a v4 checkpoint to v5.

    :param checkpoint: The checkpoint to update.
    """
    checkpoint["model_data"]["dataset_info"]._atomic_types = list(
        checkpoint["model_data"]["dataset_info"]._atomic_types
    )


def model_update_v5_v6(checkpoint: dict) -> None:
    """
    Update a v5 checkpoint to v6.

    :param checkpoint: The checkpoint to update.
    """
    if not checkpoint["best_model_state_dict"]:
        checkpoint["best_model_state_dict"] = checkpoint["model_state_dict"]


###########################
# TRAINER #################
###########################


def trainer_update_v1_v2(checkpoint: dict) -> None:
    """
    Update a v1 Trainer checkpoint to v2.

    :param checkpoint: The checkpoint to update.
    """
    checkpoint["train_hypers"]["scheduler_factor"] = checkpoint["train_hypers"].get(
        "scheduler_factor", 0.5
    )


def trainer_update_v2_v3(checkpoint: dict) -> None:
    """
    Update a v2 Trainer checkpoint to v3.

    :param checkpoint: The checkpoint to update.
    """
    checkpoint["best_epoch"] = checkpoint.get("best_epoch")


def trainer_update_v3_v4(checkpoint: dict) -> None:
    """
    Update a v3 Trainer checkpoint to v4.

    :param checkpoint: The checkpoint to update.
    """
    old_loss_hypers = checkpoint["train_hypers"]["loss"].copy()
    dataset_info = checkpoint["model_data"]["dataset_info"]
    new_loss_hypers = {}

    for target_name in dataset_info.targets.keys():
        new_loss_hypers[target_name] = {
            "type": old_loss_hypers["type"],
            "weight": old_loss_hypers["weights"].get(target_name, 1.0),
            "reduction": old_loss_hypers["reduction"],
            "sliding_factor": old_loss_hypers.get("sliding_factor", None),
        }
    checkpoint["train_hypers"]["loss"] = new_loss_hypers


def trainer_update_v4_v5(checkpoint: dict) -> None:
    """
    Update a v4 Trainer checkpoint to v5.

    :param checkpoint: The checkpoint to update.
    """
    raise ValueError(
        "In order to use this checkpoint, you need metatrain 2025.10 or earlier. "
        "You can install it with `pip install metatrain==2025.10`."
    )


def trainer_update_v5_v6(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 5 to version 6.

    :param checkpoint: The checkpoint to update.
    """
    # num_workers=0 means that the main process will do the data loading, which is
    # equivalent to not setting it (this was the behavior before v6)
    checkpoint["train_hypers"]["num_workers"] = 0
