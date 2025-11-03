import torch


class FakeScaleShift(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 1.0
        self.shift = 0.0

    def forward(self, x: torch.Tensor, head: torch.Tensor):
        return x
