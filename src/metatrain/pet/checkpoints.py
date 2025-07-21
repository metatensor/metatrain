def update_v1_v2(state_dict):
    # This if-statement is necessary to handle cases when
    # best_model_state_dict and model_state_dict are the same.
    # In that case, the both are updated within the first call of
    # this function in the PET.update_checkpoint() method.
    if "additive_models.0.model.type_to_index" not in state_dict:
        state_dict["additive_models.0.model.type_to_index"] = state_dict.pop(
            "additive_models.0.type_to_index"
        )
