def update_v1_v2(state_dict):
    state_dict["additive_models.0.model.type_to_index"] = state_dict.pop(
        "additive_models.0.type_to_index"
    )
