"""Checkpoint upgrade helpers for the LOREM architecture."""

import torch


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


def model_update_v2_v3(checkpoint: dict) -> None:
    """
    Update model checkpoint from version 2 to version 3.

    The embedded composition model and scaler now use
    ``metatensor.torch.learn.nn.Module`` with ``register_buffer`` instead of
    manually tracked buffers.

    :param checkpoint: The checkpoint to update.
    """
    for key in ["model_state_dict", "best_model_state_dict"]:
        if (state_dict := checkpoint.get(key)) is None:
            continue

        # If both model_state_dict and best_model_state_dict point to the same
        # dict, the upgrade was already applied in the first iteration.
        if "_mts_helper" in state_dict:
            continue

        reference = next(v for v in state_dict.values() if isinstance(v, torch.Tensor))
        empty_tensor = torch.zeros(0, dtype=reference.dtype, device=reference.device)

        state_dict["_mts_helper"] = empty_tensor
        state_dict["_extra_state"] = {}
