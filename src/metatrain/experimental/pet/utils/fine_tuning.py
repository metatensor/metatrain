import torch


def get_fine_tuning_weights_l2_loss(model, pretrained_weights, loss_weight):
    reg_loss = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad:
            reg_loss += loss_weight * torch.norm(param - pretrained_weights[name]) ** 2
    return reg_loss
