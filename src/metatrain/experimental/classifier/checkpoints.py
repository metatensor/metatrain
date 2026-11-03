import torch

from metatrain.utils.io import _ckpt_from_arch_ckpt


def model_update_v1_v2(checkpoint: dict) -> None:
    """
    Update a v1 checkpoint to v2.

    :param checkpoint: The checkpoint to update.
    """
    checkpoint["wrapped_model_checkpoint"] = _ckpt_from_arch_ckpt(
        checkpoint["wrapped_model_checkpoint"]
    )
    checkpoint["model_state_dict"] = checkpoint.pop("state_dict")


def model_update_v2_v3(checkpoint: dict) -> None:
    """
    Update model checkpoint from version 2 to version 3.

    :param checkpoint: The checkpoint to update.
    """
    state_dict = checkpoint["model_state_dict"]

    # If both model_state_dict and best_model_state_dict point to the same
    # dict, the upgrade was already applied in the first iteration.
    if "_mts_helper" in state_dict:
        return
    reference = next(iter(state_dict.values()))
    empty_tensor = torch.zeros(0, dtype=reference.dtype, device=reference.device)

    state_dict["_mts_helper"] = empty_tensor
    state_dict["_extra_state"] = {}
