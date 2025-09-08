import torch

class NodeEncoder(torch.nn.Module):
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
    ):
        super().__init__()

        self.center_encoder = torch.nn.Embedding(
            num_embeddings=n_species, embedding_dim=hidden_size
        )

        self.momenta_encoder = torch.nn.Sequential(
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

        self.compressor = torch.nn.Linear(
            in_features=2*hidden_size, out_features=hidden_size, bias=False
        )

    def forward(
        self,
        features: torch.Tensor,
        momenta: torch.Tensor,
    ):
        # Encode centers
        center_features = self.center_encoder(features)

        # Encode momenta
        momenta_features = self.momenta_encoder(momenta)

        # Concatenate
        encoded_features = torch.cat(
            [center_features, momenta_features], dim=-1
        )

        # Compress
        compressed_features = self.compressor(encoded_features)

        return compressed_features