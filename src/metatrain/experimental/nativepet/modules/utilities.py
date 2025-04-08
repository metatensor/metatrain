import torch


def cutoff_func(grid: torch.Tensor, r_cut: float, delta: float):
    mask_bigger = grid >= r_cut
    mask_smaller = grid <= r_cut - delta
    grid = (grid - r_cut + delta) / delta
    f = 0.5 + 0.5 * torch.cos(torch.pi * grid)

    f[mask_bigger] = 0.0
    f[mask_smaller] = 1.0
    return f


class DummyModule(torch.nn.Module):
    """Dummy torch module to make torchscript happy.
    This model should never be run"""

    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, x) -> torch.Tensor:
        raise RuntimeError("This model should never be run")
