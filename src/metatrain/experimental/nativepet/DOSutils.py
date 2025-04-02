import torch


def get_dynamic_shift_agnostic_mse(predictions, targets, cutoff_mask, return_shift = False):
    # dx is hardcoded for now
    if predictions.shape[1] < targets.shape[1]:
        smaller = predictions
        bigger = targets
    else:
        smaller = targets
        bigger = predictions

    bigger_unfolded = bigger.unfold(1, smaller.shape[1], 1)
    smaller_expanded = smaller[:, None, :]
    delta = smaller_expanded - bigger_unfolded
    # Weibin's addition - assumes prediction is bigger than target
    dynamic_delta = delta * cutoff_mask.unsqueeze(dim=1) 
    device = predictions.device
    losses = torch.trapezoid(dynamic_delta * dynamic_delta, dx = 0.05, dim=2)
    front_tail = torch.cumulative_trapezoid(predictions**2, dx = 0.05, dim = 1)
    shape_difference = predictions.shape[1] - targets.shape[1]
    additional_error = torch.hstack([torch.zeros(len(predictions), device = device).reshape(-1,1), front_tail[:,:shape_difference]])
    total_losses = losses + additional_error
    final_loss, shift = torch.min(total_losses, dim=1)
    result = torch.mean(final_loss)
    if return_shift:
        return result, shift
    else:
        return result