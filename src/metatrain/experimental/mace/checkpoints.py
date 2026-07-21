from metatrain.utils.scaler.checkpoints import update_per_property_scales


###########################
# MODEL ###################
###########################


def model_update_v1_v2(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 1 to version 2.

    :param checkpoint: The checkpoint to update.
    """
    # In v1 there was a huge bug where `edge_irreps` was passed
    # to MACE even if it was None. Here we keep this functionality
    # to update from v1 to v2, although people should really just
    # retrain, because the v1 situation was very bad.
    if checkpoint["model_data"]["hypers"]["edge_irreps"] is None:
        checkpoint["model_data"]["hypers"]["edge_irreps"] = ""


def model_update_v2_v3(checkpoint: dict) -> None:
    """
    Update a v2 checkpoint to v3.

    :param checkpoint: The checkpoint to update.
    """
    update_per_property_scales(checkpoint)


def model_update_v3_v4(checkpoint: dict) -> None:
    """
    Update a v3 checkpoint to v4.

    :param checkpoint: The checkpoint to update.
    """
    if "mace_head_name" not in checkpoint["model_data"]["hypers"]:
        checkpoint["model_data"]["hypers"]["mace_head_name"] = None


###########################
# TRAINER #################
###########################


def trainer_update_v1_v2(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 1 to version 2.

    :param checkpoint: The checkpoint to update.
    """
    checkpoint["train_hypers"]["batch_atom_bounds"] = [None, None]


def trainer_update_v2_v3(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 2 to version 3.

    :param checkpoint: The checkpoint to update.
    """
    checkpoint["train_hypers"]["max_atoms_per_batch"] = None
    checkpoint["train_hypers"]["min_atoms_per_batch"] = 0


def trainer_update_v3_v4(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 3 to version 4.

    The ``batch_atom_bounds`` field has been removed from the trainer schema
    (max-atom packing is now done by the sampler, as with ``max_atoms_per_batch``).
    ``batch_atom_bounds`` was already inert whenever ``max_atoms_per_batch`` was
    set (packing disabled the post-hoc filter), so its bounds are only
    translated into the equivalent ``max_atoms_per_batch`` / ``min_atoms_per_batch``
    sampler settings when they were the mechanism actually in effect; otherwise
    the field is simply dropped.

    :param checkpoint: The checkpoint to update.
    """
    train_hypers = checkpoint["train_hypers"]
    min_bound, max_bound = train_hypers.pop("batch_atom_bounds", [None, None])
    if train_hypers.get("max_atoms_per_batch") is None:
        if max_bound is not None:
            train_hypers["max_atoms_per_batch"] = max_bound
        if min_bound is not None:
            train_hypers["min_atoms_per_batch"] = min_bound
