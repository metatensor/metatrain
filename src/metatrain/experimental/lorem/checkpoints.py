"""Checkpoint upgrade helpers for the LOREM architecture."""

import torch

from metatrain.composition.checkpoints import (
    model_update_v1_v2 as composition_update_v1_v2,
)
from metatrain.scaler.checkpoints import (
    model_update_v1_v2 as scaler_update_v1_v2,
)


def model_update_v1_v2(checkpoint: dict) -> None:
    """
    Update model checkpoint from version 1 to version 2.

    The embedded composition model and scaler now use
    ``metatensor.torch.learn.nn.Module`` with ``register_buffer`` instead of
    manually tracked buffers.

    :param checkpoint: The checkpoint to update.
    """
    composition_update_v1_v2(checkpoint, prefix="additive_models.0.")
    scaler_update_v1_v2(checkpoint, prefix="scaler.")
    for key in ["model_state_dict", "best_model_state_dict"]:
        if (state_dict := checkpoint.get(key)) is None:
            continue

        # If both model_state_dict and best_model_state_dict point to the same
        # dict, the upgrade was already applied in the first iteration.
        if "_mts_helper" in state_dict:
            continue

        dummy_buffer = state_dict["additive_models.0.dummy_buffer"]
        empty_tensor = torch.zeros(
            0, dtype=dummy_buffer.dtype, device=dummy_buffer.device
        )

        state_dict["_mts_helper"] = empty_tensor
        state_dict["_extra_state"] = {}
