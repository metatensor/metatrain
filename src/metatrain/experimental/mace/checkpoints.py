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
