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


def model_update_v2_v3(checkpoint: dict) -> None:
    """
    Update a v2 checkpoint to v3.

    :param checkpoint: The checkpoint to update.
    """
    # state_dict renamed to model_state_dict, best_model_state_dict added with the
    # new training procedure for the ensemble by backpropagation
    checkpoint["model_state_dict"] = checkpoint.pop("state_dict")
    checkpoint["best_model_state_dict"] = None
    # changed format for ensemble weights (only do energy for simplicity)
    t = checkpoint["model_state_dict"].pop("energy_ensemble_weights").T
    checkpoint["model_state_dict"]["llpr_ensemble_layers.energy.weight"] = t
    # added num_ensemble_members to hypers (only do energy for simplicity)
    num_members = t.shape[0]
    checkpoint["model_data"]["hypers"]["num_ensemble_members"] = {"energy": num_members}
    # trainer is v1
    checkpoint["trainer_ckpt_version"] = 1
    # we set the following to None and the user will probably get errors if they're
    # accessed from a restart exercise (which would be useless anyway as there was
    # no ensemble training by backpropagation before this version)
    checkpoint["epoch"] = None
    checkpoint["optimizer_state_dict"] = None
    checkpoint["scheduler_state_dict"] = None
    checkpoint["best_epoch"] = None
    checkpoint["best_metric"] = None
    checkpoint["best_optimizer_state_dict"] = None


def trainer_update_v1_v2(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 1 to version 2.

    :param checkpoint: The checkpoint to update.
    """
    # Added distributed training hyperparameters
    if "train_hypers" in checkpoint:
        checkpoint["train_hypers"]["distributed"] = False
        checkpoint["train_hypers"]["distributed_port"] = 39591


def trainer_update_v2_v3(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 2 to version 3.

    :param checkpoint: The checkpoint to update.
    """
    if "train_hypers" in checkpoint:
        checkpoint["train_hypers"]["batch_atom_bounds"] = [None, None]
