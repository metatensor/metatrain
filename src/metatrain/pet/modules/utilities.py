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

    # Use torch.where instead of boolean indexing for compile compatibility
    # Clamp to avoid numerical issues at boundaries
    scaled_clamped = torch.clamp(scaled_values, eps, 1.0 - eps)
    bump_values = 0.5 * (1 + torch.tanh(1 / torch.tan(torch.pi * scaled_clamped)))

    # Combine conditions with torch.where
    f = torch.where(
        scaled_values <= 0.0,
        torch.ones_like(scaled_values),
        torch.where(scaled_values >= 1.0, torch.zeros_like(scaled_values), bump_values),
    )
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

    cosine_values = 0.5 + 0.5 * torch.cos(torch.pi * scaled_values)

    f = torch.where(
        scaled_values <= 0.0,
        torch.ones_like(scaled_values),
        torch.where(
            scaled_values >= 1.0, torch.zeros_like(scaled_values), cosine_values
        ),
    )
    return f


class DummyModule(torch.nn.Module):
    """Dummy torch module to make torchscript happy.
    This model should never be run"""

    def __init__(self) -> None:
        super(DummyModule, self).__init__()
        # Register a dummy parameter so the module has something
        self.register_buffer("_dummy", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return zeros instead of raising - torch.compile doesn't like exceptions
        return torch.zeros_like(x)
