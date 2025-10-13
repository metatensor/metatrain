import torch
import torch.nn as nn

from metatrain.experimental.flashmd.modules.encoder import NodeEncoder
from metatrain.pet.modules.transformer import Transformer
from metatrain.pet.modules.utilities import DummyModule


class CartesianTransformer(torch.nn.Module):
    """
    A custom transformer adapted to work with FlashMD inputs. These use momenta in
    addition to atomic types as node features.
    """

    def __init__(
        self,
        hypers,
        d_model: int,
        n_head: int,
        dim_feedforward: int,
        n_layers: int,
        dropout: float,
        n_atomic_species: int,
        is_first,
    ):
        super(CartesianTransformer, self).__init__()
        self.is_first = is_first
        self.cutoff = float(hypers["cutoff"])
        self.cutoff_width = float(hypers["cutoff_width"])
        self.trans = Transformer(
            d_model=d_model,
            num_layers=n_layers,
            n_heads=n_head,
            dim_node_features=d_model,
            dim_feedforward=dim_feedforward,
            norm="LayerNorm",
            activation="SiLU",
            transformer_type="PostLN",
        )

        self.edge_embedder = nn.Linear(4, d_model)

        if not is_first:
            n_merge = 3
        else:
            n_merge = 2

        self.compress = nn.Sequential(
            nn.Linear(n_merge * d_model, d_model),
            torch.nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.neighbor_embedder = DummyModule()  # for torchscript
        if not is_first:
            self.neighbor_embedder = nn.Embedding(n_atomic_species + 1, d_model)

        self.node_encoder = NodeEncoder(n_atomic_species + 1, d_model)

    def forward(
        self,
        input_messages: torch.Tensor,
        element_indices_nodes: torch.Tensor,
        momenta: torch.Tensor,
        element_indices_neighbors: torch.Tensor,
        edge_vectors: torch.Tensor,
        padding_mask: torch.Tensor,
        edge_distances: torch.Tensor,
        cutoff_factors: torch.Tensor,
        use_manual_attention: bool,
    ):
        node_embeddings = self.node_encoder(element_indices_nodes, momenta)
        edge_embeddings = [edge_vectors, edge_distances[:, :, None]]
        edge_embeddings = torch.cat(edge_embeddings, dim=2)
        edge_embeddings = self.edge_embedder(edge_embeddings)

        if not self.is_first:
            neighbor_elements_embedding = self.neighbor_embedder(
                element_indices_neighbors
            )
            edge_tokens = torch.cat(
                [edge_embeddings, neighbor_elements_embedding, input_messages], dim=2
            )
        else:
            neighbor_elements_embedding = torch.empty(
                0, device=edge_vectors.device, dtype=edge_vectors.dtype
            )  # for torch script
            edge_tokens = torch.cat([edge_embeddings, input_messages], dim=2)

        edge_tokens = self.compress(edge_tokens)

        padding_mask_with_central_token = torch.ones(
            padding_mask.shape[0], dtype=torch.bool, device=padding_mask.device
        )
        total_padding_mask = torch.cat(
            [padding_mask_with_central_token[:, None], padding_mask], dim=1
        )

        cutoff_subfactors = torch.ones(
            padding_mask.shape[0], device=padding_mask.device
        )
        cutoff_factors = torch.cat([cutoff_subfactors[:, None], cutoff_factors], dim=1)
        cutoff_factors[~total_padding_mask] = 0.0

        cutoff_factors = cutoff_factors[:, None, :]
        cutoff_factors = cutoff_factors.repeat(1, cutoff_factors.shape[2], 1)

        initial_num_tokens = edge_vectors.shape[1]
        max_num_tokens = input_messages.shape[1]

        output_node_embeddings, output_edge_embeddings = self.trans(
            node_embeddings[:, None, :],
            edge_tokens[:, :max_num_tokens, :],
            cutoff_factors=cutoff_factors[
                :, : (max_num_tokens + 1), : (max_num_tokens + 1)
            ],
            use_manual_attention=use_manual_attention,
        )
        if max_num_tokens < initial_num_tokens:
            padding = torch.zeros(
                output_edge_embeddings.shape[0],
                initial_num_tokens - max_num_tokens,
                output_edge_embeddings.shape[2],
                device=output_edge_embeddings.device,
            )
            output_edge_embeddings = torch.cat([output_edge_embeddings, padding], dim=1)

        output_node_embeddings = output_node_embeddings.squeeze(1)
        return output_node_embeddings, output_edge_embeddings
