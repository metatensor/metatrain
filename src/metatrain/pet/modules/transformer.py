from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .utilities import DummyModule


AVAILABLE_NORMALIZATIONS = ["LayerNorm", "RMSNorm"]
AVAILABLE_TRANSFORMER_TYPES = ["PostLN", "PreLN"]
AVAILABLE_ACTIVATIONS = ["SiLU", "SwiGLU"]


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, activation, dropout=0.0):
        super().__init__()

        # Check if activation is "swiglu" string
        if activation.lower() == "swiglu":
            # SwiGLU mode: single projection produces both "value" and "gate"
            self.w_in = nn.Linear(d_model, 2 * dim_feedforward)
            self.w_out = nn.Linear(dim_feedforward, d_model)
            self.activation = None
            self.is_swiglu = True
        else:
            # Standard mode: regular activation function
            self.w_in = nn.Linear(d_model, dim_feedforward)
            self.w_out = nn.Linear(dim_feedforward, d_model)
            self.activation = getattr(F, activation.lower())
            self.is_swiglu = False

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.is_swiglu:
            # SwiGLU activation: split into value and gate
            v, g = self.w_in(x).chunk(2, dim=-1)
            x = v * torch.sigmoid(g)
            x = self.dropout(x)
            x = self.w_out(x)
            x = self.dropout(x)
        else:
            # Standard activation
            x = self.w_in(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.w_out(x)
            x = self.dropout(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, total_dim, num_heads, epsilon=1e-15):
        super(AttentionBlock, self).__init__()

        self.input_linear = nn.Linear(total_dim, 3 * total_dim)
        self.output_linear = nn.Linear(total_dim, total_dim)

        self.num_heads = num_heads
        self.epsilon = epsilon

        if total_dim % num_heads != 0:
            raise ValueError("total dimension is not divisible by the number of heads")
        self.head_dim = total_dim // num_heads

    def forward(self, x, cutoff_factors: torch.Tensor, use_manual_attention: bool):
        initial_shape = x.shape
        x = self.input_linear(x)
        x = x.reshape(
            initial_shape[0], initial_shape[1], 3, self.num_heads, self.head_dim
        )
        x = x.permute(2, 0, 3, 1, 4)

        queries, keys, values = x[0], x[1], x[2]
        attn_weights = torch.clamp(cutoff_factors[:, None, :, :], self.epsilon)
        attn_weights = torch.log(attn_weights)
        if use_manual_attention:
            x = manual_attention(queries, keys, values, attn_weights)
        else:
            x = torch.nn.functional.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=attn_weights,
            )
        x = x.transpose(1, 2).reshape(initial_shape)
        x = self.output_linear(x)
        return x


class TransformerLayer(torch.nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dim_feedforward=512,
        dropout=0.0,
        norm="LayerNorm",
        activation="SiLU",
        transformer_type="PostLN",
    ):
        super(TransformerLayer, self).__init__()
        self.attention = AttentionBlock(d_model, n_heads)
        self.transformer_type = transformer_type
        self.d_model = d_model
        norm_class = getattr(nn, norm)
        self.norm_attention = norm_class(d_model)
        self.norm_mlp = norm_class(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mlp = FeedForward(d_model, dim_feedforward, activation, dropout)

    def _forward_pre_ln_impl(
        self,
        tokens: torch.Tensor,
        cutoff_factors: torch.Tensor,
        use_manual_attention: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = tokens + self.dropout(
            self.attention(
                self.norm_attention(tokens), cutoff_factors, use_manual_attention
            )
        )
        tokens = tokens + self.mlp(self.norm_mlp(tokens))
        output_node_embeddings, output_edge_embeddings = torch.split(
            tokens, [1, tokens.shape[1] - 1], dim=1
        )
        return output_node_embeddings, output_edge_embeddings

    def _forward_post_ln_impl(
        self,
        tokens: torch.Tensor,
        cutoff_factors: torch.Tensor,
        use_manual_attention: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.norm_attention(
            tokens
            + self.dropout(self.attention(tokens, cutoff_factors, use_manual_attention))
        )
        tokens = self.norm_mlp(tokens + self.mlp(tokens))
        output_node_embeddings, output_edge_embeddings = torch.split(
            tokens, [1, tokens.shape[1] - 1], dim=1
        )
        return output_node_embeddings, output_edge_embeddings

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        cutoff_factors: torch.Tensor,
        use_manual_attention: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.cat([node_embeddings, edge_embeddings], dim=1)
        if self.transformer_type == "PostLN":
            node_embeddings, edge_embeddings = self._forward_post_ln_impl(
                tokens, cutoff_factors, use_manual_attention
            )
        if self.transformer_type == "PreLN":
            node_embeddings, edge_embeddings = self._forward_pre_ln_impl(
                tokens, cutoff_factors, use_manual_attention
            )
        return node_embeddings, edge_embeddings


class Transformer(torch.nn.Module):
    def __init__(
        self,
        d_model,
        num_layers,
        n_heads,
        dim_feedforward=512,
        dropout=0.0,
        norm="LayerNorm",
        activation="SiLU",
        transformer_type="PostLN",
    ):
        super(Transformer, self).__init__()
        if norm not in AVAILABLE_NORMALIZATIONS:
            raise ValueError(
                f"Unknown normalization flag: {norm}. "
                f"Please choose from: {AVAILABLE_NORMALIZATIONS}"
            )
        norm_class = getattr(nn, norm)

        if transformer_type not in AVAILABLE_TRANSFORMER_TYPES:
            raise ValueError(
                f"Unknown transformer flag: {transformer_type}. "
                f"Please choose from: {AVAILABLE_TRANSFORMER_TYPES}"
            )
        self.transformer_type = transformer_type

        if activation not in AVAILABLE_ACTIVATIONS:
            raise ValueError(
                f"Unknown activation flag: {activation}. "
                f"Please choose from: {AVAILABLE_ACTIVATIONS}"
            )

        self.final_norm = DummyModule()  # for torchscript
        if transformer_type == "PreLN":
            self.final_norm = norm_class(d_model)

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    norm=norm,
                    activation=activation,
                    transformer_type=transformer_type,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        cutoff_factors: torch.Tensor,
        use_manual_attention: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            node_embeddings, edge_embeddings = layer(
                node_embeddings, edge_embeddings, cutoff_factors, use_manual_attention
            )
        if self.transformer_type == "PreLN":
            edge_embeddings = self.final_norm(edge_embeddings)
        return node_embeddings, edge_embeddings


class CartesianTransformer(torch.nn.Module):
    def __init__(
        self,
        hypers,
        d_model: int,
        n_head: int,
        dim_feedforward: int,
        n_layers: int,
        dropout: float,
        norm: str,
        activation: str,
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
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm=norm,
            activation=activation,
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

        self.node_embedder = nn.Embedding(n_atomic_species + 1, d_model)

    def forward(
        self,
        input_messages: torch.Tensor,
        element_indices_nodes: torch.Tensor,
        element_indices_neighbors: torch.Tensor,
        edge_vectors: torch.Tensor,
        padding_mask: torch.Tensor,
        edge_distances: torch.Tensor,
        cutoff_factors: torch.Tensor,
        use_manual_attention: bool,
    ):
        node_embeddings = self.node_embedder(element_indices_nodes)
        edge_embeddings = [edge_vectors, edge_distances[:, :, None]]
        edge_embeddings = torch.cat(edge_embeddings, dim=2)
        edge_embeddings = self.edge_embedder(edge_embeddings)

        if not self.is_first:
            neighbor_elements_embeddings = self.neighbor_embedder(
                element_indices_neighbors
            )
            edge_tokens = torch.cat(
                [edge_embeddings, neighbor_elements_embeddings, input_messages], dim=2
            )
        else:
            neighbor_elements_embeddings = torch.empty(
                0, device=edge_vectors.device, dtype=edge_vectors.dtype
            )  # for torch script
            edge_tokens = torch.cat([edge_embeddings, input_messages], dim=2)

        edge_tokens = self.compress(edge_tokens)
        # tokens = torch.cat([node_elements_embedding[:, None, :], tokens], dim=1)

        padding_mask_with_central_token = torch.ones(
            padding_mask.shape[0], dtype=torch.bool, device=padding_mask.device
        )
        total_padding_mask = torch.cat(
            [padding_mask_with_central_token[:, None], padding_mask], dim=1
        )

        cutoff_subfactors = torch.ones(
            padding_mask.shape[0],
            dtype=cutoff_factors.dtype,
            device=padding_mask.device,
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


def manual_attention(q, k, v, attn_mask):
    # needed for double backward (training with conservative forces)
    attention_weights = (
        torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
    ) + attn_mask
    attention_weights = attention_weights.softmax(dim=-1)
    attention_output = torch.matmul(attention_weights, v)
    return attention_output
