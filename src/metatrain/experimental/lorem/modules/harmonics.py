"""Real spherical harmonics and per-degree norms used by the LOREM port."""

from typing import List

import torch


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


def _real_spherical_harmonics(vectors: torch.Tensor, l_max: int) -> torch.Tensor:
    """Real spherical harmonics, ``m = -ℓ … +ℓ``, for ``l_max <= 2``."""
    n_lm = (l_max + 1) * (l_max + 1)
    if vectors.shape[0] == 0:
        return vectors.new_zeros((0, n_lm))
    if l_max > 2:
        raise RuntimeError(
            "Analytic spherical harmonics only support max_degree <= 2. "
            "Install sphericart-torch for higher degrees."
        )

    radius = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True).clamp(min=1.0e-12)
    x = vectors[:, 0:1] / radius
    y = vectors[:, 1:2] / radius
    z = vectors[:, 2:3] / radius

    y00 = torch.full_like(x, 0.28209479177387814)
    parts: List[torch.Tensor] = [y00]
    if l_max >= 1:
        c1 = 0.4886025119029199
        parts.extend([c1 * y, c1 * z, c1 * x])
    if l_max >= 2:
        c2 = 1.0925484305920792
        c20 = 0.31539156525252005
        c22 = 0.5462742152960396
        parts.extend(
            [
                c2 * x * y,
                c2 * y * z,
                c20 * (3.0 * z * z - 1.0),
                c2 * x * z,
                c22 * (x * x - y * y),
            ]
        )
    return torch.cat(parts, dim=-1)


class _AnalyticSphericalHarmonics(torch.nn.Module):
    """TorchScript-friendly real spherical harmonics for ``l_max <= 2``."""

    def __init__(self, l_max: int) -> None:
        super().__init__()
        self.l_max = l_max

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        return _real_spherical_harmonics(vectors, self.l_max)


class _SphericartWrapper(torch.nn.Module):
    """Wrap ``sphericart.torch.SphericalHarmonics.compute`` as ``forward``."""

    def __init__(self, calculator: torch.nn.Module) -> None:
        super().__init__()
        self.calculator = calculator

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        return self.calculator.compute(vectors)
