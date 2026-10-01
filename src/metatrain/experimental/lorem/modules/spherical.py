"""Normalization and per-degree norms of the spherical features.

sphericart returns orthonormal real spherical harmonics. LOREM uses Racah
(Schmidt semi-) normalization, as lorem-jax does. The component order stays
``m = -ℓ … +ℓ`` in each degree.
"""

import math
from typing import List

import torch


def to_racah(spherical: torch.Tensor, max_degree: int) -> torch.Tensor:
    """Convert orthonormal real SH to Racah (e3x default).

    Each ℓ-block is multiplied by :math:`\\sqrt{4\\pi / (2\\ell+1)}`.
    """
    parts: List[torch.Tensor] = []
    for ell in range(max_degree + 1):
        scale = math.sqrt(4.0 * math.pi / (2.0 * float(ell) + 1.0))
        parts.append(spherical[:, ell * ell : (ell + 1) * (ell + 1)] * scale)
    return torch.cat(parts, dim=-1)


def _safe_vector_norm(x: torch.Tensor, dim: int, eps: float = 1.0e-12) -> torch.Tensor:
    """``vector_norm`` with a finite gradient when ``x`` is exactly zero."""
    return torch.sqrt(x.pow(2).sum(dim=dim) + eps)


def _degree_norms(spherical: torch.Tensor, max_degree: int) -> torch.Tensor:
    """Per-degree norms, each scaled by ``(2ℓ+1)^{1/4}``.

    :param spherical: ``(n_atoms, (max_degree+1)**2, n_features)``
    :return: ``(n_atoms, (max_degree+1) * n_features)``
    """
    parts: List[torch.Tensor] = []
    for ell in range(max_degree + 1):
        chunk = spherical[:, ell * ell : (ell + 1) * (ell + 1), :]
        factor = (2.0 * float(ell) + 1.0) ** 0.25
        parts.append(factor * _safe_vector_norm(chunk, dim=1))
    return torch.cat(parts, dim=-1)
