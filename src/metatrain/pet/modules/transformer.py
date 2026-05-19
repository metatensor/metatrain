from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .utilities import DummyModule


class FeedForward(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()

        # HACK: override dim_feedforward so it's always 2x the model dimension
        dim_feedforward = 2 * d_model


        # SwiGLU mode: single projection produces both "value" and "gate"
        self.w_in = nn.Linear(d_model, 2 * dim_feedforward)
        self.w_out = nn.Linear(dim_feedforward, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation: split into value and gate
        v, g = self.w_in(x).chunk(2, dim=-1)
        x = v * torch.sigmoid(g)
        x = self.w_out(x)
        return x


class AttentionBlock(nn.Module):
    """
    Multi-head attention block.

    :param total_dim: The total dimension of the input and output tensors.
    :param num_heads: The number of attention heads.
    :param temperature: An additional scaling factor for attention scores.
           This is combined with the standard scaling by the square root of
           the head dimension.
    :param epsilon: A small value to avoid division by zero.
    """

    def __init__(
        self,
        total_dim: int,
        num_heads: int,
        temperature: float,
        epsilon: float = 1e-15,
    ) -> None:
        super(AttentionBlock, self).__init__()

        self.input_linear = nn.Linear(total_dim, 3 * total_dim)
        self.output_linear = nn.Linear(total_dim, total_dim)

        self.num_heads = num_heads
        self.epsilon = epsilon
        self.temperature = temperature
        if total_dim % num_heads != 0:
            raise ValueError("total dimension is not divisible by the number of heads")
        self.head_dim = total_dim // num_heads

    def forward(
        self, x: torch.Tensor, log_cutoff_factors: torch.Tensor, use_manual_attention: bool
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
        attn_weights = log_cutoff_factors[:, None, :, :]

        if use_manual_attention:
            x = manual_attention(queries, keys, values, attn_weights, self.temperature)
        else:
            x = torch.nn.functional.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=attn_weights,
                scale=1.0 / (self.head_dim**0.5 * self.temperature),
            )
        x = x.transpose(1, 2).reshape(initial_shape)
        x = self.output_linear(x)
        return x


class TransformerLayer(torch.nn.Module):
    """
    Single layer of a Transformer.

    :param d_model: The dimension of the model.
    :param n_heads: The number of attention heads.
    :param dim_node_features: The dimension of the node features.
    :param dim_feedforward: The dimension of the feedforward network.
    :param norm: The normalization type, either "LayerNorm" or "RMSNorm".
    :param activation: The activation function, either "SiLU" or "SwiGLU".
    :param transformer_type: The type of transformer, either "PostLN" or "PreLN".
    :param temperature: An additional scaling factor for attention scores.
    """

    def __init__(
        self,
        d_triplet: int,
        d_edge: int,
        d_node: int,
        n_heads: int,
        temperature: float = 1.0,
    ) -> None:
        super(TransformerLayer, self).__init__()
        self.attention = AttentionBlock(d_triplet, n_heads, temperature)
        self.d_triplet = d_triplet
        self.norm_attention = torch.nn.RMSNorm(d_triplet)
        self.norm_triplet = torch.nn.RMSNorm(d_triplet)
        self.triplet_mlp = FeedForward(d_triplet)

        self.node_contraction = nn.Linear(d_node, d_triplet)
        self.node_expansion = nn.Linear(d_triplet, d_node)
        self.norm_node = torch.nn.RMSNorm(d_node)
        self.node_mlp = FeedForward(d_node)


        self.edge_contraction = nn.Linear(d_edge, d_triplet)
        self.edge_expansion = nn.Linear(d_triplet, d_edge)
        self.norm_edge = torch.nn.RMSNorm(d_edge)
        self.edge_mlp = FeedForward(d_edge)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        triplet_embeddings: torch.Tensor,
        log_cutoff_factors: torch.Tensor,
        use_manual_attention: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single Transformer layer.

        :param node_embeddings: The input node embeddings, of shape
            (batch_size, d_model)
        :param edge_embeddings: The input edge embeddings, of shape
            (batch_size, seq_length, d_model)
        :param log_cutoff_factors: The cutoff factors for the edges, of shape
            (batch_size, seq_length, seq_length)
        :param use_manual_attention: Whether to use the manual attention implementation
            (which supports double backward, needed for training with conservative
            forces), or the built-in PyTorch attention (which does not support double
            backward).
        :return: A tuple containing:
            - The output node embeddings, of shape (batch_size, d_model)
            - The output edge embeddings, of shape (batch_size, seq_length, d_model)
        """
        edge_shape = edge_embeddings.shape[1]
        triplet_shape = triplet_embeddings.shape[1]
        input_node_embeddings = self.node_contraction(node_embeddings)
        input_edge_embeddings = self.edge_contraction(edge_embeddings)
        tokens = torch.cat([input_node_embeddings, input_edge_embeddings, triplet_embeddings], dim=1)
        new_tokens = self.attention(
            self.norm_attention(tokens), log_cutoff_factors, use_manual_attention
        )
        output_node_embeddings, output_edge_embeddings, output_triplet_embeddings = torch.split(
            new_tokens, [1, edge_shape, triplet_shape], dim=1
        )
        
        output_node_embeddings = node_embeddings + self.node_expansion(
            output_node_embeddings
        )
        output_node_embeddings = output_node_embeddings + self.node_mlp(
            self.norm_node(output_node_embeddings)
        )

        output_edge_embeddings = edge_embeddings + self.edge_expansion(
            output_edge_embeddings
        )
        output_edge_embeddings = output_edge_embeddings + self.edge_mlp(
            self.norm_edge(output_edge_embeddings)
        )

        output_triplet_embeddings = triplet_embeddings + output_triplet_embeddings
        output_triplet_embeddings = output_triplet_embeddings + self.triplet_mlp(
            self.norm_triplet(output_triplet_embeddings)
        )

        return output_node_embeddings, output_edge_embeddings, output_triplet_embeddings


class Transformer(torch.nn.Module):
    """
    Transformer implementation.

    :param d_model: The dimension of the model.
    :param num_layers: The number of transformer layers.
    :param n_heads: The number of attention heads.
    :param dim_node_features: The dimension of the node features.
    :param dim_feedforward: The dimension of the feedforward network.
    :param norm: The normalization type, either "LayerNorm" or "RMSNorm".
    :param activation: The activation function, either "SiLU" or "SwiGLU".
    :param transformer_type: The type of transformer, either "PostLN" or "PreLN".
    :param attention_temperature: The temperature scaling factor for attention
        scores. This is combined with the standard scaling by the square root of
        the head dimension.
    """

    def __init__(
        self,
        d_triplet: int,
        d_edge: int,
        d_node: int,
        num_layers: int,
        n_heads: int,
        attention_temperature: float = 1.0,
    ) -> None:
        super(Transformer, self).__init__()

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_triplet=d_triplet,
                    d_edge=d_edge,
                    d_node=d_node,
                    n_heads=n_heads,
                    temperature=attention_temperature,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        triplet_embeddings: torch.Tensor,
        log_cutoff_factors: torch.Tensor,
        use_manual_attention: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Transformer.

        :param node_embeddings: The input node embeddings, of shape
            (batch_size, d_model)
        :param edge_embeddings: The input edge embeddings, of shape
            (batch_size, seq_length, d_model)
        :param log_cutoff_factors: The cutoff factors for the edges, of shape
            (batch_size, seq_length, seq_length)
        :param use_manual_attention: Whether to use the manual attention implementation
            (which supports double backward, needed for training with conservative
            forces), or the built-in PyTorch attention (which does not support double
            backward).
        :return: A tuple containing:
            - The output node embeddings, of shape (batch_size, d_model)
            - The output edge embeddings, of shape (batch_size, seq_length, d_model)
        """
        for layer in self.layers:
            node_embeddings, edge_embeddings, triplet_embeddings = layer(
                node_embeddings, edge_embeddings, triplet_embeddings, log_cutoff_factors, use_manual_attention
            )
        return node_embeddings, edge_embeddings, triplet_embeddings


class CartesianTransformer(torch.nn.Module):
    """
    Cartesian Transformer implementation for handling 3D coordinates.

    :param cutoff: The cutoff distance for neighbor interactions.
    :param cutoff_width: The width of the cutoff function.
    :param d_model: The dimension of the model.
    :param n_head: The number of attention heads.
    :param dim_node_features: The dimension of the node features.
    :param dim_feedforward: The dimension of the feedforward network.
    :param n_layers: The number of transformer layers.
    :param norm: The normalization type, either "LayerNorm" or "RMSNorm".
    :param activation: The activation function, either "SiLU" or "SwiGLU".
    :param attention_temperature: The temperature scaling factor for attention scores.
    :param transformer_type: The type of transformer, either "PostLN" or "PreLN".
    :param n_atomic_species: The number of atomic species.
    :param is_first: Whether this is the first transformer in the model.
    """

    def __init__(
        self,
        d_triplet: int,
        d_edge: int,
        d_node: int,
        n_head: int,
        n_layers: int,
        attention_temperature: float,
    ) -> None:
        super(CartesianTransformer, self).__init__()
        self.trans = Transformer(
            d_triplet=d_triplet,
            d_edge=d_edge,
            d_node=d_node,
            num_layers=n_layers,
            n_heads=n_head,
            attention_temperature=attention_temperature,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        triplet_features: torch.Tensor,
        log_cutoff_factors_edges: torch.Tensor,
        log_cutoff_factors_triplets: torch.Tensor,
        use_manual_attention: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the CartesianTransformer.

        :param input_node_embeddings: The input node embeddings, of shape
            (n_nodes, d_model)
        :param input_messages: The input messages to the transformer, of shape
            (n_nodes, max_num_neighbors, d_model)
        :param element_indices_neighbors: The atomic species of the neighboring atoms,
            of shape (n_nodes, max_num_neighbors)
        :param edge_vectors: The cartesian edge vectors between the central atoms and
            their neighbors, of shape (n_nodes, max_num_neighbors, 3)
        :param padding_mask: A padding mask indicating which neighbors are real, and
            which are padded, of shape (n_nodes, max_num_neighbors)
        :param edge_distances: The distances between the central atoms and their
            neighbors, of shape (n_nodes, max_num_neighbors)
        :param log_cutoff_factors: The cutoff factors for the edges, of shape
            (n_nodes, max_num_neighbors)
        :param use_manual_attention: Whether to use the manual attention implementation
            (which supports double backward, needed for training with conservative
            forces), or the built-in PyTorch attention (which does not support double
            backward).
        :return: A tuple containing:
            - The output node embeddings, of shape (n_nodes, d_model)
            - The output edge embeddings, of shape (n_nodes, max_num_neighbors, d_model)
        """

        log_cutoff_factors = torch.cat([
            torch.zeros(node_features.shape[0], 1, device=node_features.device, dtype=node_features.dtype),
            log_cutoff_factors_edges,
            log_cutoff_factors_triplets,
        ], dim=1)
        log_cutoff_factors = log_cutoff_factors[:, None, :]
        log_cutoff_factors = log_cutoff_factors.repeat(1, log_cutoff_factors.shape[2], 1)

        output_node_embeddings, output_edge_embeddings, output_triplet_embeddings = self.trans(
            node_features[:, None, :],
            edge_features,
            triplet_features,
            log_cutoff_factors=log_cutoff_factors,
            use_manual_attention=use_manual_attention,
        )

        output_node_embeddings = output_node_embeddings.squeeze(1)
        return output_node_embeddings, output_edge_embeddings, output_triplet_embeddings


def manual_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Implements the attention operation manually, using basic PyTorch operations.
    We need it because the built-in PyTorch attention does not support double backward,
    which is needed when training with conservative forces.

    :param q: The queries
    :param k: The keys
    :param v: The values
    :param attn_mask: The attention mask
    :param temperature: An additional scaling factor for attention scores.
    :return: The result of the attention operation
    """
    attention_weights = (
        torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5 * temperature)
    ) + attn_mask
    attention_weights = attention_weights.softmax(dim=-1)
    attention_output = torch.matmul(attention_weights, v)
    return attention_output
