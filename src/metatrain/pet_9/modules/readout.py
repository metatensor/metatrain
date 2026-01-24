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
        num_experts: int = 1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.expert_embedding = torch.nn.Embedding(
            num_atomic_types, num_experts
        )
        self.linear = torch.nn.ModuleList(
            [
                torch.nn.Linear(feature_dim, output_dim, bias=bias)
                for _ in range(self.num_experts)
            ]
        )

    def forward(
        self,
        batch_species_indices: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        
        expert_weights = torch.softmax(
            self.expert_embedding(batch_species_indices), dim=-1
        ).unsqueeze(1)
        if features.dim() == 3:
            expert_weights = expert_weights.unsqueeze(1)

        expert_outputs = torch.stack(
            [
                linear(features)
                for linear in self.linear
            ],
            dim=-1,
        )

        return torch.sum(
            expert_outputs * expert_weights, dim=-1
        )