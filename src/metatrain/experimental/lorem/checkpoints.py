"""Checkpoint upgrade helpers for the LOREM architecture."""


def model_update_v1_v2(checkpoint: dict) -> None:
    """
    Update a v1 checkpoint to v2.

    It removes the additive models and scaler from the model checkpoint,
    as this is now handled by the MetatrainModel wrapper.

    :param checkpoint: The checkpoint to update.
    """
    removed_prefixes = (
        "additive_models.",
        "scaler.",
    )
    for key in ["model_state_dict", "best_model_state_dict"]:
        if (state_dict := checkpoint.get(key)) is not None:
            for k in list(state_dict):
                for prefix in removed_prefixes:
                    if k.startswith(prefix):
                        state_dict.pop(k)
