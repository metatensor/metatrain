import torch

from metatrain.scaler.checkpoints import update_per_property_scales
from metatrain.utils.data.atomic_basis_helpers import densify_atomic_basis_dataset_info


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
    Update a v2 checkpoint to v3.

    Adds ``finetune_config`` to the state dicts (it was not present in v2).

    :param checkpoint: The checkpoint to update.
    """
    for key in ["model_state_dict", "best_model_state_dict"]:
        if (state_dict := checkpoint.get(key)) is not None:
            if "finetune_config" not in state_dict:
                state_dict["finetune_config"] = {}


def model_update_v3_v4(checkpoint: dict) -> None:
    """
    Update a v3 checkpoint to v4.

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

    # Before this update, the model stored the dataset_info in a sparse format
    # for atomic basis targets but worked in the dense format. After this update,
    # the model works in whatever format the dataset_info is stored in,
    # so to keep the checkpoint compatible we need to densify the stored dataset_info.
    checkpoint["model_data"]["dataset_info"] = densify_atomic_basis_dataset_info(
        checkpoint["model_data"]["dataset_info"]
    )


def model_update_v4_v5(checkpoint: dict) -> None:
    """
    Update model checkpoint from version 4 to version 5.

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


#############################
# TRAINER ###################
#############################


def trainer_update_v1_v2(checkpoint: dict) -> None:
    """
    Update trainer checkpoint from version 1 to version 2.

    :param checkpoint: The checkpoint to update.
    """
    checkpoint["train_hypers"]["max_atoms_per_batch"] = None
    checkpoint["train_hypers"]["min_atoms_per_batch"] = 0


def trainer_update_v2_v3(checkpoint: dict) -> None:
    """
    Update a v2 trainer checkpoint to v3.

    Adds empty ``finetune`` hypers if not present.

    :param checkpoint: The checkpoint to update.
    """
    if "finetune" not in checkpoint["train_hypers"]:
        checkpoint["train_hypers"]["finetune"] = {
            "read_from": None,
            "method": "full",
            "config": {},
            "inherit_heads": {},
        }


def trainer_update_v3_v4(checkpoint: dict) -> None:
    """
    Deprecate the ``distributed`` hyperparameter.

    So that it doesn't show up in the restarting options.

    :param checkpoint: The checkpoint to update.
    """
    train_hypers = checkpoint["train_hypers"]
    if "distributed" in train_hypers and train_hypers["distributed"] is None:
        train_hypers.pop("distributed")
