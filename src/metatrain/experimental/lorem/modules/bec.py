"""Born effective charges (``lorem.LoremBEC`` / ``PerParticleTensorPredictor``).

Maps spherical node features to a per-atom 3×3 Cartesian tensor (APT / BEC)
with the paper's CG reconstruction, then enforces the acoustic sum rule
(zero net charge per structure) the same way ``LoremBEC.predict`` does.
"""

from typing import List

import torch

from .clebsch_gordan import ClebschGordanReal
from .tensor_dense import TensorDense


class BornEffectiveChargeHead(torch.nn.Module):
    """Per-atom 3×3 BEC / APT from spherical features.

    ``lorem-jax`` ``PerParticleTensorPredictor``: feature-wise Dense + SiLU,
    then ``TensorDense(features=1, max_degree=2)``, then the ``1 ⊗ 1 → L≤2``
    Clebsch-Gordan reconstruction to a Cartesian 3×3.
    """

    def __init__(self, in_features: int, in_max_degree: int) -> None:
        super().__init__()
        if in_max_degree < 2:
            raise ValueError(
                f"Born effective charges require max_degree >= 2 (got {in_max_degree})."
            )
        self.in_max_degree = int(in_max_degree)
        self.hidden = torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features),
            torch.nn.SiLU(),
            torch.nn.Linear(in_features, in_features),
            torch.nn.SiLU(),
            torch.nn.Linear(in_features, in_features),
        )
        self.tensor_dense = TensorDense(
            in_features=in_features,
            out_features=1,
            in_max_degree=in_max_degree,
            out_max_degree=2,
            include_pseudotensors=False,
        )
        # e3x.so3.clebsch_gordan(1, 1, 2)[1:, 1:, :] — drop ℓ=0 from each
        # 1-tensor so the 9 spherical components rebuild a 3×3.
        cg = ClebschGordanReal()
        reconstruction = torch.zeros(3, 3, 9, dtype=torch.float64)
        offset = 0
        for L in (0, 1, 2):
            reconstruction[:, :, offset : offset + 2 * L + 1] = cg.get((1, 1, L))
            offset += 2 * L + 1
        self.register_buffer(
            "cartesian_reconstruction", reconstruction.to(torch.float32)
        )

    def forward(self, spherical_features: torch.Tensor) -> torch.Tensor:
        """:param spherical_features: ``(n_atoms, n_lm, n_features)``.
        :return: ``(n_atoms, 3, 3)`` Cartesian tensors (no sum rule yet).
        """
        hidden = self.hidden(spherical_features)
        spherical = self.tensor_dense(hidden)[:, :, 0]
        return torch.einsum(
            "n l, i j l -> n i j",
            spherical,
            self.cartesian_reconstruction.to(dtype=spherical.dtype),
        )


def apply_acoustic_sum_rule(apt: torch.Tensor, system_sizes: List[int]) -> torch.Tensor:
    """Subtract the per-structure mean 3×3 so each system is charge-neutral.

    :param apt: ``(n_atoms, 3, 3)``
    :param system_sizes: number of atoms in each system, in order.
    """
    if apt.shape[0] == 0:
        return apt
    parts: List[torch.Tensor] = []
    offset = 0
    for size in system_sizes:
        chunk = apt[offset : offset + size]
        if size > 0:
            chunk = chunk - chunk.mean(dim=0, keepdim=True)
        parts.append(chunk)
        offset += size
    return torch.cat(parts, dim=0)


class DummyBornEffectiveChargeHead(torch.nn.Module):
    """Placeholder when no cartesian rank-2 target is registered (TorchScript)."""

    def forward(self, spherical_features: torch.Tensor) -> torch.Tensor:
        n_atoms = spherical_features.shape[0]
        return spherical_features.new_zeros((n_atoms, 3, 3))
