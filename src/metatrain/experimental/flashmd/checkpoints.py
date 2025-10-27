def model_update_v1_v2(checkpoint: dict) -> None:
    """
    Update a v1 checkpoint to v2.

    :param checkpoint: The checkpoint to update.
    """
    # Adding the option for choosing the normalization type
    if "normalization" not in checkpoint["model_data"]["model_hypers"]:
        checkpoint["model_data"]["model_hypers"]["normalization"] = "LayerNorm"
    # Adding the option for choosing the activation function
    if "activation" not in checkpoint["model_data"]["model_hypers"]:
        checkpoint["model_data"]["model_hypers"]["activation"] = "SiLU"
    # Setting the node features dimension to be the same as d_pet if not specified
    if "d_node" not in checkpoint["model_data"]["model_hypers"]:
        checkpoint["model_data"]["model_hypers"]["d_node"] = checkpoint["model_data"][
            "model_hypers"
        ]["d_pet"]
    # Setting the default transformer type to PostLN if not specified
    if "transformer_type" not in checkpoint["model_data"]["model_hypers"]:
        checkpoint["model_data"]["model_hypers"]["transformer_type"] = "PostLN"
    if "featurizer_type" not in checkpoint["model_data"]["model_hypers"]:
        checkpoint["model_data"]["model_hypers"]["featurizer_type"] = "residual"
    for key in ["model_state_dict", "best_model_state_dict"]:
        if (state_dict := checkpoint.get(key)) is not None:
            new_state_dict = {}
            for k, v in state_dict.items():
                # Replacing the nn.Sequential MLP with a custom FeedForward module
                if ".mlp.0" in k:
                    k = k.replace(".mlp.0", ".mlp.w_in")
                if ".mlp.3" in k:
                    k = k.replace(".mlp.3", ".mlp.w_out")
                new_state_dict[k] = v
            checkpoint[key] = new_state_dict


def trainer_update_v1_v2(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 1 to version 2.

    :param checkpoint: The checkpoint to update.
    """
    # remove all entries in the loss `sliding_factor`
    old_loss_hypers = checkpoint["train_hypers"]["loss"].copy()
    dataset_info = checkpoint["model_data"]["dataset_info"]
    new_loss_hypers = {}

    for target_name in dataset_info.targets.keys():
        # retain everything except sliding_factor for each target
        new_loss_hypers[target_name] = {
            k: v
            for k, v in old_loss_hypers[target_name].items()
            if k != "sliding_factor"
        }

    checkpoint["train_hypers"]["loss"] = new_loss_hypers
