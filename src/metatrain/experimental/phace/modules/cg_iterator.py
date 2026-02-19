from typing import Dict, List

import torch

from .layers import EquivariantRMSNorm
from .layers import LinearList as Linear
from .tensor_product import tensor_product


class CGIterator(torch.nn.Module):
    """High-level CG iterator that chains multiple CG iterations."""

    def __init__(self, k_max_l, number_of_iterations):
        super().__init__()
        self.number_of_iterations = number_of_iterations

        # CG iterations
        cg_iterations = []
        for _ in range(number_of_iterations):
            cg_iterations.append(CGIteration(k_max_l))
        self.cg_iterations = torch.nn.ModuleList(cg_iterations)

    def forward(
        self, features: List[torch.Tensor], U_dict: Dict[int, torch.Tensor]
    ) -> List[torch.Tensor]:
        for iterator in self.cg_iterations:
            features = iterator(features, U_dict)
        return features


class CGIteration(torch.nn.Module):
    """A single Clebsch-Gordan-like iteration.

    Implements RMSNorm -> linear -> tensor product -> linear -> skip connection.
    """

    def __init__(self, k_max_l):
        super().__init__()
        self.linear_in = Linear(k_max_l, expansion_factor=2)
        self.rmsnorm = EquivariantRMSNorm(k_max_l)
        self.linear_out = Linear([2 * k for k in k_max_l], expansion_factor=0.5)

    def forward(
        self,
        features: List[torch.Tensor],
        U_dict: Dict[int, torch.Tensor],
    ) -> List[torch.Tensor]:
        features_in = features
        features = self.rmsnorm(features)
        features = self.linear_in(features, U_dict)
        features = tensor_product(features, features)
        features = self.linear_out(features, U_dict)
        features_out = [f1 + fo for f1, fo in zip(features_in, features, strict=True)]
        return features_out
