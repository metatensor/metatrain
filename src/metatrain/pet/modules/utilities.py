import torch
from metatensor.torch.learn.nn import Module


def cutoff_func_bump(
    values: torch.Tensor, cutoff: torch.Tensor, width: float, eps: float = 1e-6
) -> torch.Tensor:
    """
    Bump cutoff function.

    :param values: Distances at which to evaluate the cutoff function.
    :param cutoff: Cutoff radius for each node.
    :param width: Width of the cutoff region.
    :param eps: Avoid computing at values too close to the edges.
    :return: Values of the cutoff function at the specified distances.
    """

    scaled_values = (values - (cutoff - width)) / width

    mask_smaller = scaled_values <= 0.0
    mask_active = (scaled_values > 0.0) & (scaled_values < 1.0)

    f = torch.zeros_like(scaled_values)
    f[mask_active] = 0.5 * (
        1 + torch.tanh(1 / torch.tan(torch.pi * scaled_values[mask_active]))
    )
    f[mask_smaller] = 1.0

    return f


def cutoff_func_cosine(
    values: torch.Tensor, cutoff: torch.Tensor, width: float
) -> torch.Tensor:
    """
    Cosine cutoff function.

    :param values: Distances at which to evaluate the cutoff function.
    :param cutoff: Cutoff radius for each node.
    :param width: Width of the cutoff region.
    :return: Values of the cutoff function at the specified distances.
    """

    scaled_values = (values - (cutoff - width)) / width

    mask_smaller = scaled_values <= 0.0
    mask_active = (scaled_values > 0.0) & (scaled_values < 1.0)

    f = torch.zeros_like(scaled_values)

    f[mask_active] = 0.5 + 0.5 * torch.cos(torch.pi * scaled_values[mask_active])
    f[mask_smaller] = 1.0
    return f


class DummyModule(Module):
    """Dummy torch module to make torchscript happy.
    This model should never be run"""

    def __init__(self) -> None:
        super(DummyModule, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("This model should never be run")
