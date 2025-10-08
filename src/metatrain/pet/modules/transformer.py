from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .utilities import DummyModule


AVAILABLE_NORMALIZATIONS = ["LayerNorm", "RMSNorm"]
AVAILABLE_TRANSFORMER_TYPES = ["PostLN", "PreLN"]
AVAILABLE_ACTIVATIONS = ["SiLU", "SwiGLU"]


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, activation: str) -> None:
        super().__init__()

        # Check if activation is "swiglu" string
        if activation.lower() == "swiglu":
            # SwiGLU mode: single projection produces both "value" and "gate"
            self.w_in = nn.Linear(d_model, 2 * dim_feedforward)
            self.w_out = nn.Linear(dim_feedforward, d_model)
            self.activation = torch.nn.Identity()
            self.is_swiglu = True
        else:
            # Standard mode: regular activation function
            self.w_in = nn.Linear(d_model, dim_feedforward)
            self.w_out = nn.Linear(dim_feedforward, d_model)
            self.activation = getattr(F, activation.lower())
            self.is_swiglu = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_swiglu:
            # SwiGLU activation: split into value and gate
            v, g = self.w_in(x).chunk(2, dim=-1)
            x = v * torch.sigmoid(g)
            x = self.w_out(x)
        else:
            # Standard activation
            x = self.w_in(x)
            x = self.activation(x)
            x = self.w_out(x)
        return x


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
    :param dim_node_features: The dimension of the node features.
    :param dim_feedforward: The dimension of the feedforward network.
    :param norm: The normalization type, either "LayerNorm" or "RMSNorm".
    :param activation: The activation function, either "SiLU" or "SwiGLU".
    :param transformer_type: The type of transformer, either "PostLN" or "PreLN".
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_node_features: int,
        dim_feedforward: int = 512,
        norm: str = "LayerNorm",
        activation: str = "SiLU",
        transformer_type: str = "PostLN",
    ) -> None:
        super(TransformerLayer, self).__init__()
        self.attention = AttentionBlock(d_model, n_heads)
        self.transformer_type = transformer_type
        self.d_model = d_model
        norm_class = getattr(nn, norm)
        self.norm_attention = norm_class(d_model)
        self.norm_mlp = norm_class(d_model)
        self.mlp = FeedForward(d_model, dim_feedforward, activation)
        self.expanded_node_features = False
        if dim_node_features != d_model:
            self.expanded_node_features = True
            self.center_contraction = nn.Linear(dim_node_features, d_model)
            self.center_expansion = nn.Linear(d_model, dim_node_features)
            self.norm_center_features = norm_class(dim_node_features)
            self.center_mlp = FeedForward(
                dim_node_features, 2 * dim_node_features, activation
            )
        else:
            self.center_contraction = torch.nn.Identity()
            self.center_expansion = torch.nn.Identity()
            self.norm_center_features = torch.nn.Identity()
            self.center_mlp = torch.nn.Identity()

    def _forward_pre_ln_impl(
        self,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        cutoff_factors: torch.Tensor,
        use_manual_attention: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.expanded_node_features:
            input_node_embeddings = self.center_contraction(node_embeddings)
        else:
            input_node_embeddings = node_embeddings
        tokens = torch.cat([input_node_embeddings, edge_embeddings], dim=1)
        new_tokens = self.attention(
            self.norm_attention(tokens), cutoff_factors, use_manual_attention
        )
        output_node_embeddings, output_edge_embeddings = torch.split(
            new_tokens, [1, new_tokens.shape[1] - 1], dim=1
        )
        if self.expanded_node_features:
            output_node_embeddings = node_embeddings + self.center_expansion(
                output_node_embeddings
            )
            output_node_embeddings = output_node_embeddings + self.center_mlp(
                self.norm_center_features(output_node_embeddings)
            )

        output_edge_embeddings = edge_embeddings + output_edge_embeddings
        output_edge_embeddings = output_edge_embeddings + self.mlp(
            self.norm_mlp(output_edge_embeddings)
        )

        return output_node_embeddings, output_edge_embeddings

    def _forward_post_ln_impl(
        self,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        cutoff_factors: torch.Tensor,
        use_manual_attention: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.expanded_node_features:
            input_node_embeddings = self.center_contraction(node_embeddings)
        else:
            input_node_embeddings = node_embeddings
        tokens = torch.cat([input_node_embeddings, edge_embeddings], dim=1)
        tokens = self.norm_attention(
            tokens + self.attention(tokens, cutoff_factors, use_manual_attention)
        )
        tokens = self.norm_mlp(tokens + self.mlp(tokens))
        output_node_embeddings, output_edge_embeddings = torch.split(
            tokens, [1, tokens.shape[1] - 1], dim=1
        )
        if self.expanded_node_features:
            output_node_embeddings = node_embeddings + self.center_expansion(
                output_node_embeddings
            )
            output_node_embeddings = output_node_embeddings + self.center_mlp(
                self.norm_center_features(output_node_embeddings)
            )
        return output_node_embeddings, output_edge_embeddings

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        cutoff_factors: torch.Tensor,
        use_manual_attention: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single Transformer layer.

        :param node_embeddings: The input node embeddings, of shape
            (batch_size, d_model)
        :param edge_embeddings: The input edge embeddings, of shape
            (batch_size, seq_length, d_model)
        :param cutoff_factors: The cutoff factors for the edges, of shape
            (batch_size, seq_length, seq_length)
        :param use_manual_attention: Whether to use the manual attention implementation
            (which supports double backward, needed for training with conservative
            forces), or the built-in PyTorch attention (which does not support double
            backward).
        :return: A tuple containing:
            - The output node embeddings, of shape (batch_size, d_model)
            - The output edge embeddings, of shape (batch_size, seq_length, d_model)
        """
        if self.transformer_type == "PostLN":
            node_embeddings, edge_embeddings = self._forward_post_ln_impl(
                node_embeddings, edge_embeddings, cutoff_factors, use_manual_attention
            )
        if self.transformer_type == "PreLN":
            node_embeddings, edge_embeddings = self._forward_pre_ln_impl(
                node_embeddings, edge_embeddings, cutoff_factors, use_manual_attention
            )
        return node_embeddings, edge_embeddings


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
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        n_heads: int,
        dim_node_features: int,
        dim_feedforward: int = 512,
        norm: str = "LayerNorm",
        activation: str = "SiLU",
        transformer_type: str = "PostLN",
    ) -> None:
        super(Transformer, self).__init__()
        if norm not in AVAILABLE_NORMALIZATIONS:
            raise ValueError(
                f"Unknown normalization flag: {norm}. "
                f"Please choose from: {AVAILABLE_NORMALIZATIONS}"
            )

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

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_node_features=dim_node_features,
                    dim_feedforward=dim_feedforward,
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
        """
        Forward pass for the Transformer.

        :param node_embeddings: The input node embeddings, of shape
            (batch_size, d_model)
        :param edge_embeddings: The input edge embeddings, of shape
            (batch_size, seq_length, d_model)
        :param cutoff_factors: The cutoff factors for the edges, of shape
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
            node_embeddings, edge_embeddings = layer(
                node_embeddings, edge_embeddings, cutoff_factors, use_manual_attention
            )
        return node_embeddings, edge_embeddings


class CartesianTransformer(torch.nn.Module):
    """
    Cartesian Transformer implementation for handling 3D coordinates.

    :param hypers: A dictionary of hyperparameters.
    :param d_model: The dimension of the model.
    :param n_head: The number of attention heads.
    :param dim_node_features: The dimension of the node features.
    :param dim_feedforward: The dimension of the feedforward network.
    :param n_layers: The number of transformer layers.
    :param norm: The normalization type, either "LayerNorm" or "RMSNorm".
    :param activation: The activation function, either "SiLU" or "SwiGLU".
    :param transformer_type: The type of transformer, either "PostLN" or "PreLN".
    :param n_atomic_species: The number of atomic species.
    :param is_first: Whether this is the first transformer in the model.
    """

    def __init__(
        self,
        hypers: Dict[str, Any],
        d_model: int,
        n_head: int,
        dim_node_features: int,
        dim_feedforward: int,
        n_layers: int,
        norm: str,
        activation: str,
        transformer_type: str,
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
            dim_node_features=dim_node_features,
            dim_feedforward=dim_feedforward,
            norm=norm,
            activation=activation,
            transformer_type=transformer_type,
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

    def forward(
        self,
        input_node_embeddings: torch.Tensor,
        input_messages: torch.Tensor,
        element_indices_neighbors: torch.Tensor,
        edge_vectors: torch.Tensor,
        padding_mask: torch.Tensor,
        edge_distances: torch.Tensor,
        cutoff_factors: torch.Tensor,
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
        :param cutoff_factors: The cutoff factors for the edges, of shape
            (n_nodes, max_num_neighbors)
        :param use_manual_attention: Whether to use the manual attention implementation
            (which supports double backward, needed for training with conservative
            forces), or the built-in PyTorch attention (which does not support double
            backward).
        :return: A tuple containing:
            - The output node embeddings, of shape (n_nodes, d_model)
            - The output edge embeddings, of shape (n_nodes, max_num_neighbors, d_model)
        """
        node_embeddings = input_node_embeddings
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
