import torch


def cutoff_func(
    values: torch.Tensor, cutoff: torch.Tensor, width: float
) -> torch.Tensor:
    """
    Bump cutoff function.

    :param grid: Distances at which to evaluate the cutoff function.
    :param cutoff: Cutoff radius for each node.
    :param width: Width of the cutoff region.
    :return: Values of the cutoff function at the specified distances.
    """
    mask_bigger = values >= cutoff
    mask_smaller = values <= cutoff - width
    scaled_values = (values - (cutoff - width)) / width

    f = 0.5 * (1 + torch.tanh(1 / torch.tan(torch.pi * scaled_values)))
    # print("bump", values.shape, cutoff.shape, scaled_values.shape)
    # print("cutoff", cutoff, values, f[0], f[:,0])

    f[mask_bigger] = 0.0
    f[mask_smaller] = 1.0
    return f


class DummyModule(torch.nn.Module):
    """Dummy torch module to make torchscript happy.
    This model should never be run"""

    def __init__(self) -> None:
        super(DummyModule, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("This model should never be run")
