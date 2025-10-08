from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .utilities import DummyModule


class AttentionBlock(nn.Module):
    """
    Multi-head attention block.

    :param total_dim: The total dimension of the input and output tensors.
    :param num_heads: The number of attention heads.
    :param epsilon: A small value to avoid division by zero.
    """

    def __init__(self, total_dim: int, num_heads: int, epsilon: float = 1e-15) -> None:
        super(AttentionBlock, self).__init__()

        self.input_linear = nn.Linear(total_dim, 3 * total_dim)
        self.output_linear = nn.Linear(total_dim, total_dim)

        nn.init.xavier_uniform_(self.input_linear.weight)
        nn.init.constant_(self.input_linear.bias, 0.0)
        nn.init.constant_(self.output_linear.bias, 0.0)

        self.num_heads = num_heads
        self.epsilon = epsilon

        if total_dim % num_heads != 0:
            raise ValueError("total dimension is not divisible by the number of heads")
        self.head_dim = total_dim // num_heads

    def forward(
        self, x: torch.Tensor, cutoff_factors: torch.Tensor, use_manual_attention: bool
    ) -> torch.Tensor:
        """
        Forward pass for the attention block.

        :param x: The input tensor, of shape (batch_size, seq_length, total_dim)
        :param cutoff_factors: The cutoff factors for the edges, of shape
            (batch_size, seq_length, seq_length)
        :param use_manual_attention: Whether to use the manual attention implementation
            (which supports double backward, needed for training with conservative
            forces), or the built-in PyTorch attention (which does not support double
            backward).
        :return: The output tensor, of shape (batch_size, seq_length, total_dim)
        """
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
    """
    Single layer of a Transformer.

    :param d_model: The dimension of the model.
    :param n_heads: The number of attention heads.
    :param dim_feedforward: The dimension of the feedforward network.
    :param dropout: The dropout rate.
    :param activation: The activation function.
    :param transformer_type: The type of transformer, either "PostLN" or "PreLN".
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        activation: Callable = F.silu,
        transformer_type: str = "PostLN",
    ) -> None:
        super(TransformerLayer, self).__init__()
        self.attention = AttentionBlock(d_model, n_heads)

        if transformer_type not in ["PostLN", "PreLN"]:
            raise ValueError("unknown transformer type")
        self.transformer_type = transformer_type
        self.d_model = d_model
        self.norm_attention = nn.LayerNorm(d_model)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = activation

        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        cutoff_factors: torch.Tensor,
        use_manual_attention: bool,
    ) -> torch.Tensor:
        """
        Forward pass for a single Transformer layer.

        :param tokens: The input tokens to the transformer layer, of shape
            (batch_size, seq_length, d_model)
        :param cutoff_factors: The cutoff factors for the edges, of shape
            (batch_size, seq_length, seq_length)
        :param use_manual_attention: Whether to use the manual attention implementation
            (which supports double backward, needed for training with conservative
            forces), or the built-in PyTorch attention (which does not support double
            backward).
        :return: The output tokens of the transformer layer, of shape
            (batch_size, seq_length, d_model)
        """
        if self.transformer_type == "PostLN":
            tokens = self.norm_attention(
                tokens
                + self.dropout(
                    self.attention(tokens, cutoff_factors, use_manual_attention)
                )
            )
            tokens = self.norm_mlp(tokens + self.mlp(tokens))
        if self.transformer_type == "PreLN":
            tokens = tokens + self.dropout(
                self.attention(
                    self.norm_attention(tokens), cutoff_factors, use_manual_attention
                )
            )
            tokens = tokens + self.mlp(self.norm_mlp(tokens))
        return tokens


class Transformer(torch.nn.Module):
    """
    Transformer implementation.

    :param d_model: The dimension of the model.
    :param num_layers: The number of transformer layers.
    :param n_heads: The number of attention heads.
    :param dim_feedforward: The dimension of the feedforward network.
    :param dropout: The dropout rate.
    :param activation: The activation function.
    :param transformer_type: The type of transformer, either "PostLN" or "PreLN".
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        n_heads: int,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        activation: Callable = F.silu,
        transformer_type: str = "PostLN",
    ) -> None:
        super(Transformer, self).__init__()
        self.transformer_type = transformer_type

        self.final_norm = DummyModule()  # for torchscript
        if transformer_type == "PreLN":
            self.final_norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    transformer_type=transformer_type,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        tokens: torch.Tensor,
        cutoff_factors: torch.Tensor,
        use_manual_attention: bool,
    ) -> torch.Tensor:
        """
        Forward pass for the Transformer.
        :param tokens: The input tokens to the transformer, of shape
            (batch_size, seq_length, d_model)
        :param cutoff_factors: The cutoff factors for the edges, of shape
            (batch_size, seq_length, seq_length)
        :param use_manual_attention: Whether to use the manual attention implementation
            (which supports double backward, needed for training with conservative
            forces), or the built-in PyTorch attention (which does not support double
            backward).
        :return: The output tokens of the transformer, of shape
            (batch_size, seq_length, d_model)
        """
        for layer in self.layers:
            tokens = layer(tokens, cutoff_factors, use_manual_attention)
        if self.transformer_type == "PreLN":
            tokens = self.final_norm(tokens)
        return tokens


class CartesianTransformer(torch.nn.Module):
    """
    Cartesian Transformer implementation for handling 3D coordinates.

    :param hypers: A dictionary of hyperparameters.
    :param d_model: The dimension of the model.
    :param n_head: The number of attention heads.
    :param dim_feedforward: The dimension of the feedforward network.
    :param n_layers: The number of transformer layers.
    :param dropout: The dropout rate.
    :param n_atomic_species: The number of atomic species.
    :param is_first: Whether this is the first transformer in the model.
    """

    def __init__(
        self,
        hypers: Dict[str, Any],
        d_model: int,
        n_head: int,
        dim_feedforward: int,
        n_layers: int,
        dropout: float,
        n_atomic_species: int,
        is_first: bool,
    ) -> None:
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
            activation=torch.nn.SiLU(),
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the CartesianTransformer.

        :param input_messages: The input messages to the transformer, of shape
            (n_nodes, max_num_neighbors, d_model)
        :param element_indices_nodes: The atomic species of the central atoms, of shape
            (n_nodes,)
        :param element_indices_neighbors: The atomic species of the neighboring atoms,
            of shape (n_nodes, max_num_neighbors)
        :param edge_vectors: The cartesian edge vectors between the central atoms and
            their neighbors, of shape (n_nodes, max_num_neighbors, 3)
        :param padding_mask: A padding mask indicating which neighbors are real, and
            which are padded, of shape (n_nodes, max_num_neighbors)
        :param edge_distances: The distances between the central atoms and their
            neighbors, of shape (n_nodes, max_num_neighbors)
        :param cutoff_factors: The cutoff factors for the edges, of shape
            (n_nodes, max_num_neighbors)
        :param use_manual_attention: Whether to use the manual attention implementation
            (which supports double backward, needed for training with conservative
            forces), or the built-in PyTorch attention (which does not support double
            backward).
        :return: A tuple with the output node embeddings of shape (n_nodes, d_pet)
        """
        node_elements_embedding = self.node_embedder(element_indices_nodes)
        edge_embedding = [edge_vectors, edge_distances[:, :, None]]
        edge_embedding = torch.cat(edge_embedding, dim=2)
        edge_embedding = self.edge_embedder(edge_embedding)

        if not self.is_first:
            neighbor_elements_embedding = self.neighbor_embedder(
                element_indices_neighbors
            )
            tokens = torch.cat(
                [edge_embedding, neighbor_elements_embedding, input_messages], dim=2
            )
        else:
            neighbor_elements_embedding = torch.empty(
                0, device=edge_vectors.device, dtype=edge_vectors.dtype
            )  # for torch script
            tokens = torch.cat([edge_embedding, input_messages], dim=2)

        tokens = self.compress(tokens)
        tokens = torch.cat([node_elements_embedding[:, None, :], tokens], dim=1)

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

        output_messages = self.trans(
            tokens[:, : (max_num_tokens + 1), :],
            cutoff_factors=cutoff_factors[
                :, : (max_num_tokens + 1), : (max_num_tokens + 1)
            ],
            use_manual_attention=use_manual_attention,
        )
        if max_num_tokens < initial_num_tokens:
            padding = torch.zeros(
                output_messages.shape[0],
                initial_num_tokens - max_num_tokens,
                output_messages.shape[2],
                device=output_messages.device,
            )
            output_messages = torch.cat([output_messages, padding], dim=1)

        output_node_embeddings = output_messages[:, 0, :]
        output_edge_embeddings = output_messages[:, 1:, :]
        return output_node_embeddings, output_edge_embeddings


def manual_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor
) -> torch.Tensor:
    """
    Implements the attention operation manually, using basic PyTorch operations.
    We need it because the built-in PyTorch attention does not support double backward,
    which is needed when training with conservative forces.

    :param q: The queries
    :param k: The keys
    :param v: The values
    :param attn_mask: The attention mask
    :return: The result of the attention operation
    """
    attention_weights = (
        torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
    ) + attn_mask
    attention_weights = attention_weights.softmax(dim=-1)
    attention_output = torch.matmul(attention_weights, v)
    return attention_output
