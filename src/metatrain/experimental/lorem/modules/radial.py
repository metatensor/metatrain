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
    # ``pow`` at base 0 (r == 0 or r == cutoff, also after clamping) has NaN
    # second derivatives (0 * 0**-1), which breaks force training for pairs
    # landing exactly on the cutoff. Evaluate on a safe x in the open interval
    # and select the exact endpoint values with ``where``, which routes
    # gradients instead of multiplying them (0 * NaN would still be NaN).
    strict = ((r > 0.0) & (r < cutoff)).unsqueeze(-1)
    x_safe = torch.where(strict, x, torch.full_like(x, 0.5))
    values = (
        binomial.to(device=r.device, dtype=r.dtype)
        * x_safe.pow(k)
        * (1.0 - x_safe).pow(float(num - 1) - k)
    )
    edges = torch.zeros_like(values)
    edges[:, 0] = (r == 0.0).to(r.dtype)
    edges[:, -1] = edges[:, -1] + (r == cutoff).to(r.dtype)
    return torch.where(strict, values, edges)
