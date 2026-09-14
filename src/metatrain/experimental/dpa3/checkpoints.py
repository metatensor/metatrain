import torch

from metatrain.composition.checkpoints import (
    model_update_v1_v2 as composition_update_v1_v2,
)
from metatrain.scaler.checkpoints import (
    model_update_v1_v2 as scaler_update_v1_v2,
)
from metatrain.scaler.checkpoints import update_per_property_scales


###########################
# MODEL ###################
###########################


def model_update_v1_v2(checkpoint: dict) -> None:
    """
    Update a v1 checkpoint to v2.

    :param checkpoint: The checkpoint to update.
    """
    update_per_property_scales(checkpoint)


def model_update_v2_v3(checkpoint: dict) -> None:
    """
    Update model checkpoint from version 2 to version 3.

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


###########################
# TRAINER #################
###########################


def trainer_update_v1_v2(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 1 to version 2.

    :param checkpoint: The checkpoint to update.
    """
    checkpoint["train_hypers"]["max_atoms_per_batch"] = None
    checkpoint["train_hypers"]["min_atoms_per_batch"] = 0


def trainer_update_v2_v3(checkpoint: dict) -> None:
    """
    Deprecate the ``distributed`` hyperparameter.

    So that it doesn't show up in the restarting options.

    :param checkpoint: The checkpoint to update.
    """
    train_hypers = checkpoint["train_hypers"]
    if "distributed" in train_hypers and train_hypers["distributed"] is None:
        train_hypers.pop("distributed")
