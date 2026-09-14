import torch



def model_update_v1_v2(checkpoint: dict) -> None:
    """
    Update model checkpoint from version 1 to version 2.

    :param checkpoint: The checkpoint to update.
    """
    state_dict = checkpoint["state_dict"]

    # If both model_state_dict and best_model_state_dict point to the same
    # dict, the upgrade was already applied in the first iteration.
    if "_mts_helper" in state_dict:
        return
    dummy_buffer = next(iter(state_dict.values()))
    empty_tensor = torch.zeros(
        0, dtype=dummy_buffer.dtype, device=dummy_buffer.device
    )
    empty_tensor = torch.zeros(
        0, dtype=dummy_buffer.dtype, device=dummy_buffer.device
    )

    state_dict["_mts_helper"] = empty_tensor
    state_dict["_extra_state"] = {}
