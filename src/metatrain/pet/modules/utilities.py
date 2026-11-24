import torch


def cutoff_func_bump(
    values: torch.Tensor, cutoff: torch.Tensor, width: float
) -> torch.Tensor:
    """
    Bump cutoff function.

    The 1e-6 margin from the boundary avoids numerical instabilities
    in the region where the functions transitions with C^inf smoothness
    from 1 and to 0.

    The function is ``0.5 * (1 + tanh(cot(pi * x)))``, evaluated as the equivalent
    ``sigmoid(2 * cot(pi * x))``. This avoids the cancellation in ``1 + tanh`` when
    the cutoff factor is small, which matters because PET uses its logarithm as an
    attention bias.

    :param values: Distances at which to evaluate the cutoff function.
    :param cutoff: Cutoff radius for each node.
    :param width: Width of the cutoff region.
    :return: Values of the cutoff function at the specified distances.
    """

    scaled_values = (values - (cutoff - width)) / width
    clamped = scaled_values.clamp(1e-6, 1.0 - 1e-6)
    return torch.sigmoid(2 / torch.tan(torch.pi * clamped))


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
    clamped = scaled_values.clamp(0.0, 1.0)
    return 0.5 * (1 + torch.cos(torch.pi * clamped))


class DummyModule(torch.nn.Module):
    """Dummy torch module to make torchscript happy.
    This model should never be run"""

    def __init__(self) -> None:
        super(DummyModule, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("This model should never be run")
