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


def trainer_update_v1_v2(state_dict):
    if "scheduler_factor" not in state_dict["train_hypers"]:
        state_dict["train_hypers"]["scheduler_factor"] = 0.5


def model_update_v2_v3(state_dict):
    if state_dict is not None:
        if "train_hypers" in state_dict:
            finetune_config = state_dict["train_hypers"].get("finetune", {})
        else:
            finetune_config = {}
        state_dict["finetune_config"] = finetune_config
