import torch


class FakeScaleShift(torch.nn.Module):
    """A no-op scale-shift module used to replace MACE's scale_shift block.

    This module is used when removing the scale and shift from a pretrained
    MACE model, as these operations are handled by metatrain's Scaler and
    CompositionModel classes instead.
    """

    def __init__(self):
        super().__init__()
        self.scale = 1.0
        self.shift = 0.0

    def forward(self, x: torch.Tensor, head: torch.Tensor):
        return x
