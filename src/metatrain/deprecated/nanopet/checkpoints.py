###########################
# MODEL ###################
###########################


def model_update_v1_v2(checkpoint):
    for key in ["model_state_dict", "best_model_state_dict"]:
        if (state_dict := checkpoint.get(key)) is not None:
            if "additive_models.0.model.type_to_index" not in state_dict:
                state_dict["additive_models.0.model.type_to_index"] = state_dict.pop(
                    "additive_models.0.type_to_index"
                )


###########################
# TRAINER #################
###########################


def trainer_update_v1_v2(checkpoint):
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
