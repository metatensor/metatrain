"""Bernstein radial basis used by the short-range density.

This is e3x / lorem-jax ``basic_bernstein`` (no learnable weights).
"""

import torch


def binomial_row(num: int) -> torch.Tensor:
    """``C(num - 1, k)`` for ``k = 0 … num - 1``."""
    if num < 1:
        raise ValueError(f"num_radial must be >= 1, got {num}")
    coeffs = torch.ones(num, dtype=torch.float64)
    for k in range(1, num):
        coeffs[k] = coeffs[k - 1] * float(num - k) / float(k)
    return coeffs


def bernstein_basis(
    r: torch.Tensor, cutoff: float, binomial: torch.Tensor
) -> torch.Tensor:
    """e3x ``basic_bernstein``: :math:`B_k^{K-1}(r / l)` on ``[0, cutoff]``.

    :param r: Distances ``(n_edges,)``.
    :param cutoff: Bernstein limit ``l``.
    :param binomial: ``C(K-1, k)`` of length ``K``.
    :return: ``(n_edges, K)``, zero outside ``[0, cutoff]``.
    """
    num = binomial.shape[0]
    k = torch.arange(num, device=r.device, dtype=r.dtype)
    x = (r / cutoff).unsqueeze(-1)
    inside = (r >= 0.0) & (r <= cutoff)
    x_clamped = x.clamp(min=0.0, max=1.0)
    values = (
        binomial.to(device=r.device, dtype=r.dtype)
        * x_clamped.pow(k)
        * (1.0 - x_clamped).pow(float(num - 1) - k)
    )
    return values * inside.to(r.dtype).unsqueeze(-1)
