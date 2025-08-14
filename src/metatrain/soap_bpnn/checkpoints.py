def model_update_v1_v2(checkpoint):
    for key in ["model_state_dict", "best_model_state_dict"]:
        if (state_dict := checkpoint.get(key)) is not None:
            state_dict["additive_models.0.model.type_to_index"] = state_dict.pop(
                "additive_models.0.type_to_index"
            )


def model_update_v2_v3(checkpoint):
    checkpoint["epoch"] = checkpoint.get("epoch")
    checkpoint["best_epoch"] = checkpoint.get("best_epoch")

    if checkpoint["best_model_state_dict"] is not None:
        checkpoint["best_model_state_dict"] = checkpoint.get("best_model_state_dict")


def trainer_update_v1_v2(checkpoint):
    checkpoint["best_epoch"] = checkpoint.get("best_epoch")
