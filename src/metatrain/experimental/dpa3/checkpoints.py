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
