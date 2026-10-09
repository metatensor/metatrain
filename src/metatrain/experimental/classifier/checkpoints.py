from metatrain.utils.io import _ckpt_from_arch_ckpt

def model_update_v1_v2(checkpoint: dict) -> None:
    """
    Update a v4 checkpoint to v5.

    :param checkpoint: The checkpoint to update.
    """
    checkpoint["wrapped_model_checkpoint"] = _ckpt_from_arch_ckpt(
        checkpoint["wrapped_model_checkpoint"]
    )
    checkpoint["model_state_dict"] = checkpoint.pop("state_dict")