import torch


def model_update_v1_v2(checkpoint: dict) -> None:
    """
    Update a v1 PositionAdditive checkpoint to v2.

    :param checkpoint: The checkpoint to update.
    """
    for key in ["model_state_dict", "best_model_state_dict"]:
        if (state_dict := checkpoint.get(key)) is None:
            continue

        if "_mts_helper" in state_dict:
            continue

        # The v1 state dict is empty, so there is no tensor to take a dtype from
        state_dict["_mts_helper"] = torch.zeros(0)
        state_dict["_extra_state"] = {}
