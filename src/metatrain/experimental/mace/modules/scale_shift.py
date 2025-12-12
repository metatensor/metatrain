import torch


class FakeScaleShift(torch.nn.Module):
    """This is a module with the same interface as mace's ScaleShift module,
    but that does nothing.

    It is used as a replacement when we want to remove MACE's scale and shift
    (e.g., when loading a pretrained MACE model), which will likely be replaced
    by metatrain's Scaler and CompositionModel classes.
    """

    def __init__(self):
        super().__init__()
        self.scale = torch.tensor(1.0)
        self.shift = torch.tensor(0.0)

    def forward(self, x: torch.Tensor, head: torch.Tensor):
        return x
