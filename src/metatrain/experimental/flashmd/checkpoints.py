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
