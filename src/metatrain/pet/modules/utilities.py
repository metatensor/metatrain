import torch


def cutoff_func(grid: torch.Tensor, r_cut: torch.Tensor, delta: float) -> torch.Tensor:
    """
    Cosine cutoff function.

    :param grid: Distances at which to evaluate the cutoff function.
    :param r_cut: Cutoff radius for each node.
    :param delta: Width of the cutoff region.
    :return: Values of the cutoff function at the specified distances.
    """
    mask_bigger = grid >= r_cut
    mask_smaller = grid <= r_cut - delta
    grid = (grid - r_cut + delta) / delta
    f = 0.5 + 0.5 * torch.cos(torch.pi * grid)

    f[mask_bigger] = 0.0
    f[mask_smaller] = 1.0
    return f


def step_characteristic_function(
    values: torch.Tensor, threshold: torch.Tensor, width: float
) -> torch.Tensor:
    """Compute the step characteristic function values.
    :param values: Input values (torch.Tensor).
    :param threshold: Threshold value (torch.Tensor).
    :param width: Width parameter (float).

    :return: Step characteristic function values (torch.Tensor).
    """
    x = (values - threshold) / width
    return 0.5 * (1.0 - torch.tanh(x))


def smooth_delta_function(
    values: torch.Tensor, center: torch.Tensor, width: float
) -> torch.Tensor:
    """Compute the smooth delta function values.
    :param values: Input values (torch.Tensor).
    :param center: Center value (torch.Tensor).
    :param width: Width parameter (float).

    :return: Smooth delta function values (torch.Tensor).
    """
    x = (values - center) / width
    return torch.exp(-(x**2)) / (width * torch.sqrt(torch.tensor(torch.pi)))


class DummyModule(torch.nn.Module):
    """Dummy torch module to make torchscript happy.
    This model should never be run"""

    def __init__(self) -> None:
        super(DummyModule, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("This model should never be run")
