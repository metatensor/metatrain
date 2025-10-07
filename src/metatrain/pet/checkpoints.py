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


def model_update_v2_v3(checkpoint):
    for key in ["model_state_dict", "best_model_state_dict"]:
        if (state_dict := checkpoint.get(key)) is not None:
            if "train_hypers" in state_dict:
                finetune_config = state_dict["train_hypers"].get("finetune", {})
            else:
                finetune_config = {}
            state_dict["finetune_config"] = finetune_config


def model_update_v3_v4(checkpoint):
    checkpoint["epoch"] = checkpoint.get("epoch")
    checkpoint["best_epoch"] = checkpoint.get("best_epoch")

    if checkpoint["best_model_state_dict"] is not None:
        checkpoint["best_model_state_dict"] = checkpoint.get("best_model_state_dict")


def model_update_v4_v5(checkpoint):
    checkpoint["model_data"]["dataset_info"]._atomic_types = list(
        checkpoint["model_data"]["dataset_info"]._atomic_types
    )


def model_update_v5_v6(checkpoint):
    if not checkpoint["best_model_state_dict"]:
        checkpoint["best_model_state_dict"] = checkpoint["model_state_dict"]


def model_update_v6_v7(checkpoint):
    # Adding the option for choosing the normalization type
    if "normalization" not in checkpoint["model_data"]["model_hypers"]:
        checkpoint["model_data"]["model_hypers"]["normalization"] = "LayerNorm"
    # Adding the option for choosing the activation function
    if "activation" not in checkpoint["model_data"]["model_hypers"]:
        checkpoint["model_data"]["model_hypers"]["activation"] = "SiLU"
    for key in ["model_state_dict", "best_model_state_dict"]:
        if (state_dict := checkpoint.get(key)) is not None:
            new_state_dict = {}
            for k, v in state_dict.items():
                # Replacing the nn.Sequential MLP with a custom FeedForward module
                if ".mlp.0" in k:
                    k = k.replace(".mlp.0", ".mlp.w_in")
                if ".mlp.3" in k:
                    k = k.replace(".mlp.3", ".mlp.w_out")
                # Moving the node embedder to a top-level PET model attribute
                if "embedding." in k:
                    k = k.replace("embedding.", "edge_embedder.")
                if ".node_embedder." in k:
                    key_content = k.split(".")
                    k = ".".join(["node_embedders", key_content[1], key_content[-1]])
                new_state_dict[k] = v
            checkpoint[key] = new_state_dict


###########################
# TRAINER #################
###########################


def trainer_update_v1_v2(checkpoint):
    checkpoint["train_hypers"]["scheduler_factor"] = checkpoint["train_hypers"].get(
        "scheduler_factor", 0.5
    )


def trainer_update_v2_v3(checkpoint):
    checkpoint["best_epoch"] = checkpoint.get("best_epoch")


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


def trainer_update_v4_v5(checkpoint):
    raise ValueError(
        "In order to use this checkpoint, you need metatrain 2025.10 or earlier. "
        "You can install it with `pip install metatrain==2025.10`."
    )
