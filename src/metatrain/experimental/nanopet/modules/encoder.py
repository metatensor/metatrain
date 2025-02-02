from typing import Dict, Optional

import torch


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
        incoming_message: bool = True,
    ):
        super().__init__()

        self.cartesian_encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=3, out_features=4 * hidden_size, bias=False),
            torch.nn.SiLU(),
            torch.nn.Linear(
                in_features=4 * hidden_size, out_features=4 * hidden_size, bias=False
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(
                in_features=4 * hidden_size, out_features=hidden_size, bias=False
            ),
        )
        self.center_encoder = torch.nn.Embedding(
            num_embeddings=n_species, embedding_dim=hidden_size
        )
        self.neighbor_encoder = torch.nn.Embedding(
            num_embeddings=n_species, embedding_dim=hidden_size
        )
        self.compressor = torch.nn.Linear(
            in_features=(4 if incoming_message else 3) * hidden_size,
            out_features=hidden_size,
            bias=False,
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        incoming_message_tensor: Optional[torch.Tensor] = None,
    ):
        # Encode cartesian coordinates
        cartesian_features = self.cartesian_encoder(features["cartesian"])

        # Encode centers
        center_features = self.center_encoder(features["center"])

        # Encode neighbors
        neighbor_features = self.neighbor_encoder(features["neighbor"])

        # Concatenate
        features_to_concatenate = [
            cartesian_features,
            center_features,
            neighbor_features,
        ]
        if incoming_message_tensor is not None:
            features_to_concatenate.append(incoming_message_tensor)
        encoded_features = torch.concatenate(features_to_concatenate, dim=-1)

        # Compress
        compressed_features = self.compressor(encoded_features)

        return compressed_features
