import torch


def cutoff_func(
    values: torch.Tensor, cutoff: torch.Tensor, width: float
) -> torch.Tensor:
    """
    Bump cutoff function.

    :param values: Distances at which to evaluate the cutoff function.
    :param cutoff: Cutoff radius for each node.
    :param width: Width of the cutoff region.
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
