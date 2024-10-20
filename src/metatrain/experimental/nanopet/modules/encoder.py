from typing import Dict

import torch


class Encoder(torch.nn.Module):

    def __init__(
        self,
        n_species: int,
        hidden_size: int,
    ):
        super().__init__()

        self.cartesian_encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=3, out_features=4 * hidden_dim, bias=False),
            torch.nn.SiLU(),
            torch.nn.Linear(in_features=4 * hidden_dim, out_features=4 * hidden_dim, bias=False),
            torch.nn.SiLU(),
            torch.nn.Linear(in_features=4 * hidden_dim, out_features=hidden_dim, bias=False)
        )
        self.center_encoder = torch.nn.Embedding(
            num_embeddings=n_species, embedding_dim=hidden_size
        )
        self.neighbor_encoder = torch.nn.Embedding(
            num_embeddings=n_species, embedding_dim=hidden_size
        )
        self.compressor = torch.nn.Linear(
            in_features=3 * hidden_size, out_features=hidden_size, bias=False
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
    ):
        # Encode cartesian coordinates
        cartesian_features = self.cartesian_encoder(features["cartesian"])

        # Encode centers
        center_features = self.center_encoder(features["center"])

        # Encode neighbors
        neighbor_features = self.neighbor_encoder(features["neighbor"])

        # Concatenate
        encoded_features = torch.concatenate(
            [cartesian_features, center_features, neighbor_features], dim=-1
        )

        # Compress
        compressed_features = self.compressor(encoded_features)

        return compressed_features
