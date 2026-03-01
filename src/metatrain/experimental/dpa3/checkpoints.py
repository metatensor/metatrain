import torch


def model_update_v1_v2(checkpoint):
    """Add the padding_mask_threshold buffer (registered in v2)."""
    threshold = torch.tensor(1e-10)
    checkpoint["model_state_dict"]["padding_mask_threshold"] = threshold
    if checkpoint["best_model_state_dict"] is not None:
        checkpoint["best_model_state_dict"]["padding_mask_threshold"] = threshold
