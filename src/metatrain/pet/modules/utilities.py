import torch


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
    clamped = scaled_values.clamp(eps, 1.0 - eps)
    return 0.5 * (1 + torch.tanh(1 / torch.tan(torch.pi * clamped)))


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
    clamped = scaled_values.clamp(eps, 1.0)
    return 0.5 * (1 + torch.cos(torch.pi * clamped))

class DummyModule(torch.nn.Module):
    """Dummy torch module to make torchscript happy.
    This model should never be run"""

    def __init__(self) -> None:
        super(DummyModule, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("This model should never be run")
