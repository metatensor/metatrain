from typing import Dict

import torch
from sphericart.torch import SphericalHarmonics


class Encoder(torch.nn.Module):
    """
    An encoder of edges. It generates a fixed-size representation of the
    interatomic vector, the chemical element of the center and the chemical
    element of the neighbor. The representations are then concatenated and
    compressed to the initial fixed size.
    """

    def __init__(
        self,
        n_species: int,
        hidden_size: int,
        max_angular: int,
        max_radial: int,
        cutoff_radius: float,
    ):
        super().__init__()

        self.neighbor_encoder = torch.nn.Embedding(
            num_embeddings=n_species, embedding_dim=hidden_size
        )
        self.spherical_harmonics = SphericalHarmonics(l_max=max_angular)
        self.radial_mlp = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=max_radial, out_features=4 * hidden_size, bias=False
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(
                in_features=4 * hidden_size, out_features=4 * hidden_size, bias=False
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(
                in_features=4 * hidden_size, out_features=4 * hidden_size, bias=False
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(
                in_features=4 * hidden_size, out_features=hidden_size, bias=False
            ),
        )

        self.max_radial = max_radial
        self.cutoff_radius = cutoff_radius

    def forward(
        self,
        features: Dict[str, torch.Tensor],
    ):
        # Encode neighbors
        neighbor_features = self.neighbor_encoder(
            features["neighbor"]
        )  # [n_edges, hidden_size]

        # Encode angles
        spherical_harmonics = self.spherical_harmonics(
            features["cartesian"]
        )  # [n_edges, (max_angular + 1) ** 2]

        # Encode distances (Spherical Bessel with l = 0 for simplicity)
        distances = torch.sqrt(
            torch.sum(features["cartesian"] ** 2, dim=-1)
        )  # [n_edges]
        scaled_distances = torch.stack(
            [
                distances * n * torch.pi / self.cutoff_radius
                for n in range(1, self.max_radial + 1)
            ],
            dim=-1,
        )  # [n_edges, max_radial]
        radial_features = (
            torch.sin(scaled_distances) / scaled_distances
        )  # [n_edges, max_radial]
        radial_features = self.radial_mlp(radial_features)  # [n_edges, hidden_size]

        features = (
            neighbor_features.unsqueeze(2)
            * radial_features.unsqueeze(2)
            * spherical_harmonics.unsqueeze(1)
        )  # [n_edges, hidden_size, (max_angular + 1) ** 2]
        return features
