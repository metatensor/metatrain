from typing import Dict

import torch


class Encoder(torch.nn.Module):
    """
    Assemble the inputs for transformer layer: Mix geometric information
    and previous features, and add a central token.
    """

    def __init__(
        self,
        n_species: int,
        hidden_size: int,
    ):
        super().__init__()

        self.cartesian_encoder = torch.nn.Linear(
            in_features=3, out_features=hidden_size
        )

        self.mixer = torch.nn.Sequential(
            torch.nn.Linear(in_features=2 * hidden_size, out_features=hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(in_features=hidden_size, out_features=hidden_size),
        )
        self.center_encoder = torch.nn.Embedding(
            num_embeddings=n_species, embedding_dim=hidden_size
        )

    def forward(
        self,
        fixed: Dict[str, torch.Tensor],
        features: torch.Tensor,
    ):
        cartesian_features = self.cartesian_encoder(fixed["cartesian"])  # per edge

        center_features = self.center_encoder(fixed["center"])  # per node

        features = torch.concatenate([cartesian_features, features], dim=-1)
        features = self.mixer(features)

        features = torch.concatenate([features, center_features.unsqueeze(1)], dim=1)

        return features
