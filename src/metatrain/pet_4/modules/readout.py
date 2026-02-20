from typing import List
import torch

from metatomic.torch import System


class ReadoutLayer(torch.nn.Module):
    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        num_atomic_types: int,
        bias: bool = True,
    ):
        super().__init__()
        self.shift = torch.nn.Embedding(
            num_atomic_types, feature_dim
        )
        self.linear = torch.nn.Linear(feature_dim, output_dim, bias=bias)

    def forward(
        self,
        batch_species_indices: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        
        shift = self.shift(batch_species_indices)
        if features.dim() == 3:  # edge features
            shift = shift.unsqueeze(1)
        features = features + shift

        return self.linear(features)