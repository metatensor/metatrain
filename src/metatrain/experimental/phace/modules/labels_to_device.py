import torch
from metatensor.torch import Labels


def move_labels_to_device(labels: Labels, device: torch.device) -> Labels:
    if labels.device != device:
        labels = labels.to(device)
    return labels
