def model_update_v1_v2(checkpoint: dict) -> None:
    """
    Update a v1 checkpoint to v2.

    :param checkpoint: The checkpoint to update.
    """
    ensemble_sizes = {}
    for key in checkpoint["state_dict"].keys():
        if key.endswith("_ensemble_weights"):
            ensemble_sizes[key.replace("_ensemble_weights", "")] = checkpoint[
                "state_dict"
            ][key].shape[1]
    checkpoint["model_data"] = {}
    checkpoint["model_data"]["hypers"] = {
        "ensembles": {
            "num_members": ensemble_sizes,
            # correct means not needed to load the state_dict correctly
            "means": {name: [] for name in ensemble_sizes.keys()},
        },
    }
    checkpoint["model_data"]["dataset_info"] = checkpoint["wrapped_model_checkpoint"][
        "model_data"
    ]["dataset_info"]
