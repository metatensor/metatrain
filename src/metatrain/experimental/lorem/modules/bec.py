"""Born effective charges.

A per-atom 3×3 tensor from spherical features. The linear layers are
per degree, with a bias only on ``ℓ = 0``. SiLU gates every component by
the scalar channel, which is what keeps the head equivariant. The final
coupling keeps the even-parity channel, so the antisymmetric part is a
pseudovector and the tensor is unchanged by inversion. Each structure is
then shifted so its tensors sum to zero.

The Dense layers are chained (``Dense → silu → Dense → silu → Dense``).
lorem-jax 0.1.0 feeds the input to all three instead, a bug to be fixed there
(lab-cosmo/lorem-jax#38), so its BEC checkpoints give different charges here.
"""

from typing import List

import torch

from .clebsch_gordan import ClebschGordanReal, cg_phase_correction
from .tensor_dense import TensorDense, _DegreeWiseLinear


def gated_silu(x: torch.Tensor) -> torch.Tensor:
    """e3x ``silu``: ``sigmoid`` of the ``ℓ = 0`` channel multiplies ``x``.

    On a pure scalar this is ordinary SiLU. On higher degrees it is a gate,
    not an elementwise SiLU.
    """
    return torch.sigmoid(x[:, :1, :]) * x


class BornEffectiveChargeHead(torch.nn.Module):
    """Spherical features to a per-atom ``(3, 3)`` tensor."""

    def __init__(
        self, in_features: int, hidden_features: int, in_max_degree: int
    ) -> None:
        super().__init__()
        if in_max_degree < 2:
            raise ValueError(
                f"Born effective charges require max_degree >= 2 (got {in_max_degree})."
            )
        degree = int(in_max_degree)
        hidden = int(hidden_features)
        self.dense0 = _DegreeWiseLinear(int(in_features), hidden, degree, bias=True)
        self.dense1 = _DegreeWiseLinear(hidden, hidden, degree, bias=True)
        self.dense2 = _DegreeWiseLinear(hidden, hidden, degree, bias=True)
        self.tensor_dense = TensorDense(
            in_features=hidden,
            out_features=1,
            in_max_degree=degree,
            out_max_degree=2,
            use_bias=True,
            even_parity=True,
        )
        cg = ClebschGordanReal()
        reconstruction = torch.zeros(3, 3, 9, dtype=torch.float64)
        offset = 0
        for ell in (0, 1, 2):
            # e3x's Clebsch-Gordan signs, so lorem-jax weights give the same tensor
            reconstruction[:, :, offset : offset + 2 * ell + 1] = cg.get(
                (1, 1, ell)
            ) * cg_phase_correction(1, 1, ell)
            offset += 2 * ell + 1
        self.register_buffer(
            "cartesian_reconstruction", reconstruction.to(torch.float64)
        )

    def forward(self, spherical_features: torch.Tensor) -> torch.Tensor:
        hidden = gated_silu(self.dense0(spherical_features))
        hidden = gated_silu(self.dense1(hidden))
        hidden = self.dense2(hidden)
        spherical = self.tensor_dense(hidden)[:, :, 0]
        # CG input axes follow real-SH ℓ=1 order (y, z, x). Metatensor
        # components are (x, y, z).
        yzx = torch.einsum(
            "n l, i j l -> n i j",
            spherical,
            self.cartesian_reconstruction.to(dtype=spherical.dtype),
        )
        order = torch.tensor([2, 0, 1], device=yzx.device)
        return yzx.index_select(1, order).index_select(2, order)


class BecPredictor(torch.nn.Module):
    """Sum of one head per short-range stage, plus one on the long-range mix.

    ``snapshots`` is ``(n_stages, n_atoms, n_lm, features)``. The long-range
    argument is the spherical update produced by mixing Coulomb potentials
    back into the node features.
    """

    def __init__(
        self,
        n_stages: int,
        in_features: int,
        hidden_features: int,
        in_max_degree: int,
    ) -> None:
        super().__init__()
        self.stages = torch.nn.ModuleList(
            [
                BornEffectiveChargeHead(in_features, hidden_features, in_max_degree)
                for _ in range(n_stages)
            ]
        )
        self.long_range = BornEffectiveChargeHead(
            in_features, hidden_features, in_max_degree
        )

    def forward(
        self, snapshots: torch.Tensor, spherical_updates: torch.Tensor
    ) -> torch.Tensor:
        apt = self.long_range(spherical_updates)
        for index, head in enumerate(self.stages):
            apt = apt + head(snapshots[index])
        return apt


def apply_acoustic_sum_rule(apt: torch.Tensor, system_sizes: List[int]) -> torch.Tensor:
    """Subtract the per-structure mean 3×3 so each system sums to zero."""
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
