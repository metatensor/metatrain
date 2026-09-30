"""Normalization of the real spherical harmonics used by the backbone.

sphericart and the analytic fallback return 4π-normalized harmonics. LOREM
uses Racah (Schmidt semi-) normalization, as lorem-jax does. The component
order stays ``m = -ℓ … +ℓ`` in each degree.
"""

import math
from typing import List

import torch


def to_racah(spherical: torch.Tensor, max_degree: int) -> torch.Tensor:
    """Convert 4π-normalized real SH to Racah (e3x default).

    Each ℓ-block is multiplied by :math:`\\sqrt{4\\pi / (2\\ell+1)}`.
    """
    parts: List[torch.Tensor] = []
    for ell in range(max_degree + 1):
        scale = math.sqrt(4.0 * math.pi / (2.0 * float(ell) + 1.0))
        parts.append(spherical[:, ell * ell : (ell + 1) * (ell + 1)] * scale)
    return torch.cat(parts, dim=-1)
