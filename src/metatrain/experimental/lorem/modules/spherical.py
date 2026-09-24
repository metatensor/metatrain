"""Real spherical harmonics in the backbone ``m = -ℓ … +ℓ`` layout.

``orthonormal`` is the original 4π-normalized basis (sphericart / analytic).
``e3x`` rescales each ℓ-block to Racah / Schmidt semi-normalization, the
e3x ``spherical_harmonics(..., normalization='racah')`` default used by
lorem-jax. m-ordering stays ``-ℓ … +ℓ`` so Clebsch–Gordan (SOAP convention)
is unchanged; e3x cartesian order is a permutation within each block.
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
