# ===== Model checkpoint updates =====


def model_update_v1_v2(state_dict):
    # This if-statement is necessary to handle cases when
    # best_model_state_dict and model_state_dict are the same.
    # In that case, the both are updated within the first call of
    # this function in the PET.update_checkpoint() method.
    if (
        state_dict is not None
        and "additive_models.0.model.type_to_index" not in state_dict
    ):
        state_dict["additive_models.0.model.type_to_index"] = state_dict.pop(
            "additive_models.0.type_to_index"
        )


def model_update_v2_v3(state_dict):
    if state_dict is not None:
        if "train_hypers" in state_dict:
            finetune_config = state_dict["train_hypers"].get("finetune", {})
        else:
            finetune_config = {}
        state_dict["finetune_config"] = finetune_config


# ===== Trainer checkpoint updates =====


def trainer_update_v3_v4(checkpoint):
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
