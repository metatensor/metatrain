import torch


def get_fine_tuning_weights_l2_loss(model, pretrained_weights, loss_weight):
    reg_loss = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad:
            pretrained_param = pretrained_weights[name]
            assert (
                not pretrained_param.requires_grad
            )  # Pretrained params should not be updated
            reg_loss += loss_weight * torch.norm(param - pretrained_param) ** 2
    return reg_loss
