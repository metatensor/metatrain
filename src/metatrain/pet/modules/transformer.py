import warnings
from typing import Final, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .utilities import DummyModule


AVAILABLE_NORMALIZATIONS = ["LayerNorm", "RMSNorm"]
AVAILABLE_TRANSFORMER_TYPES = ["PostLN", "PreLN"]
AVAILABLE_ACTIVATIONS = ["SiLU", "SwiGLU"]

# SDPA's CUDA backward grids the attention batch (n_nodes) over gridDim.y/z, which
# overflows above this many nodes. We work around it by looping SDPA over chunks of
# the batch dimension instead.
MAX_CUDA_GRID_DIM: Final[int] = 65535


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
    :param temperature: An additional scaling factor for attention scores.
           This is combined with the standard scaling by the square root of
           the head dimension.
    :param smooth_cutoff: Whether the central token attends to its neighbors with
        weights that grow smoothly with their cutoff factors, see
        :func:`smooth_cutoff_attention`.
    :param epsilon: A small value to avoid division by zero.
    """

    def __init__(
        self,
        total_dim: int,
        num_heads: int,
        temperature: float,
        smooth_cutoff: bool = False,
        epsilon: float = 1e-15,
    ) -> None:
        super(AttentionBlock, self).__init__()

        self.input_linear = nn.Linear(total_dim, 3 * total_dim)
        self.output_linear = nn.Linear(total_dim, total_dim)

        self.num_heads = num_heads
        self.epsilon = epsilon
        self.temperature = temperature
        self.smooth_cutoff = smooth_cutoff
        if total_dim % num_heads != 0:
            raise ValueError("total dimension is not divisible by the number of heads")
        self.head_dim = total_dim // num_heads
        self.max_cuda_grid_dim = MAX_CUDA_GRID_DIM
        self._warned_cuda_grid_dim = False

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

        factors = cutoff_factors[:, None, :, :]
        excluded = factors <= 0.0

        # clamp to avoid log(0) in autograd
        attn_weights = torch.log(torch.clamp(factors, self.epsilon))

        # excluded keys get zero weight in the softmax
        attn_weights = attn_weights.masked_fill(excluded, float("-inf"))

        if self.smooth_cutoff:
            # The central token comes first. It is the one whose attention a single
            # neighbor can take over, so it is the only query that needs the smooth
            # weights. Attention is computed row by row, so its row can be done on
            # its own at O(num_keys) cost, while the other queries (still attending
            # to all the keys) go through the standard, possibly fused, attention.
            scale = 1.0 / (self.head_dim**0.5 * self.temperature)
            central_logits = scale * torch.matmul(
                queries[:, :, :1], keys.transpose(-2, -1)
            )
            central_weights = smooth_cutoff_attention(central_logits, factors[:, :, :1])
            central = torch.matmul(central_weights, values)
            others = self._attention(
                queries[:, :, 1:],
                keys,
                values,
                attn_weights[:, :, 1:],
                use_manual_attention,
            )
            x = torch.cat([central, others], dim=2)
        else:
            x = self._attention(
                queries, keys, values, attn_weights, use_manual_attention
            )

        x = x.transpose(1, 2).reshape(initial_shape)
        x = self.output_linear(x)
        return x

    def _attention(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor,
        use_manual_attention: bool,
    ) -> torch.Tensor:
        """
        Standard scaled dot-product attention with an additive mask.

        :param queries: The queries, of shape (batch_size, num_heads, num_queries,
            head_dim)
        :param keys: The keys, of shape (batch_size, num_heads, num_keys, head_dim)
        :param values: The values, of shape (batch_size, num_heads, num_keys,
            head_dim)
        :param attn_mask: The additive attention mask, broadcastable to
            (batch_size, num_heads, num_queries, num_keys)
        :param use_manual_attention: Whether to use the manual attention
            implementation, which supports double backward.
        :return: The attention output, of shape (batch_size, num_heads, num_queries,
            head_dim)
        """
        scale = 1.0 / (self.head_dim**0.5 * self.temperature)
        batch_size = queries.shape[0]
        if use_manual_attention:
            x = manual_attention(queries, keys, values, attn_mask, self.temperature)
        elif queries.requires_grad and batch_size > self.max_cuda_grid_dim:
            if not self._warned_cuda_grid_dim:
                warnings.warn(
                    "The number of nodes exceeds the CUDA grid dimension of "
                    f"{self.max_cuda_grid_dim}, which may cause an overflow "
                    "in the backward pass of the "
                    "torch.nn.functional.scaled_dot_product_attention."
                    "Falling back to a for-loop over batch chunks, which "
                    "may reduce performance. If you are running the model inference, "
                    "please consider using the LAMMPS interface with > 1 MPI rank to "
                    "enable domain decomposition. More details can be found on "
                    "https://lab-cosmo.github.io/upet/latest/usage/lammps.html ",
                    stacklevel=2,
                )
                self._warned_cuda_grid_dim = True
            outputs = []
            for start in range(0, batch_size, self.max_cuda_grid_dim):
                end = start + self.max_cuda_grid_dim
                outputs.append(
                    torch.nn.functional.scaled_dot_product_attention(
                        queries[start:end],
                        keys[start:end],
                        values[start:end],
                        attn_mask=attn_mask[start:end],
                        scale=scale,
                    )
                )
            x = torch.cat(outputs, dim=0)
        else:
            x = torch.nn.functional.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=attn_mask,
                scale=scale,
            )
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
    :param smooth_cutoff: Whether to use a smooth attention cutoff, see
        :class:`AttentionBlock`.
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
        temperature: float = 1.0,
        smooth_cutoff: bool = False,
    ) -> None:
        super(TransformerLayer, self).__init__()
        self.attention = AttentionBlock(d_model, n_heads, temperature, smooth_cutoff)
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
        else:
            output_node_embeddings = node_embeddings + output_node_embeddings

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
    :param attention_temperature: The temperature scaling factor for attention
        scores. This is combined with the standard scaling by the square root of
        the head dimension.
    :param smooth_cutoff: Whether to use a smooth attention cutoff, see
        :class:`AttentionBlock`.
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
        attention_temperature: float = 1.0,
        smooth_cutoff: bool = False,
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
                    temperature=attention_temperature,
                    smooth_cutoff=smooth_cutoff,
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
    :param smooth_attention_cutoff: Whether the cutoff function enters the attention
        in a way that avoids sharp features when an atom enters or leaves the cutoff.
        The attention that the central atom pays to a neighbor is then proportional
        to its cutoff function (see :func:`smooth_cutoff_attention`), and the cutoff
        function is not applied to the attention of a neighbor to itself.
    """

    def __init__(
        self,
        cutoff: float,
        cutoff_width: float,
        d_model: int,
        n_head: int,
        dim_node_features: int,
        dim_feedforward: int,
        n_layers: int,
        norm: str,
        activation: str,
        attention_temperature: float,
        transformer_type: str,
        n_atomic_species: int,
        is_first: bool,
        smooth_attention_cutoff: bool = False,
    ) -> None:
        super(CartesianTransformer, self).__init__()
        self.is_first = is_first
        self.smooth_attention_cutoff = smooth_attention_cutoff
        self.cutoff = cutoff
        self.cutoff_width = cutoff_width
        self.trans = Transformer(
            d_model=d_model,
            num_layers=n_layers,
            n_heads=n_head,
            dim_node_features=dim_node_features,
            dim_feedforward=dim_feedforward,
            norm=norm,
            activation=activation,
            transformer_type=transformer_type,
            attention_temperature=attention_temperature,
            smooth_cutoff=smooth_attention_cutoff,
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
            self.neighbor_embedder = nn.Embedding(n_atomic_species, d_model)

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

        # on some systems, on isolated atoms, a torchscript bug concatenates the two
        # (empty) float tensors into an int tensors, causing an error later on
        edge_embeddings = torch.cat(edge_embeddings, dim=2).to(edge_vectors.dtype)

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

        if self.smooth_attention_cutoff:
            # The cutoff factor of a neighbor controls how much the other tokens
            # attend to it, but its own token attends to itself at full weight.
            # Otherwise the features of a neighbor in the cutoff tail depend on its
            # own cutoff factor f, and since everything this token contributes to is
            # multiplied by f again, the energy picks up terms in f^2, f^3, ... that
            # vary much faster with distance than f itself. Padded tokens keep a
            # zero factor, so that they stay excluded from the attention.
            num_tokens = cutoff_factors.shape[1]
            is_self = torch.eye(
                num_tokens, dtype=torch.bool, device=cutoff_factors.device
            )
            is_self = is_self[None, :, :] & total_padding_mask[:, :, None]
            cutoff_factors = torch.where(
                is_self, torch.ones_like(cutoff_factors), cutoff_factors
            )

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


def smooth_cutoff_attention(
    logits: torch.Tensor, factors: torch.Tensor, established_power: float = 8.0
) -> torch.Tensor:
    """
    Attention weights that grow smoothly with the cutoff factors of the keys.

    Adding ``log(f)`` to the logits gives a key the weight ``f * exp(logit)`` in the
    softmax. A key whose logit is much larger than those of the other keys then takes
    most of the attention as soon as it enters the cutoff, however small ``f`` is.
    Here, the weight of a key before normalization is instead

    .. code-block:: text

        f * exp(logit) / (1 + (1 - f) * exp(logit) / established)

    where ``established`` is the total weight of the *other* keys, each counted with
    a factor ``f**established_power``, i.e. only once they are well inside the cutoff.
    For a single key in the cutoff tail, the attention it receives is then exactly
    ``f`` times what it would receive with ``f = 1``. Keys with ``f = 1`` are not
    affected.

    :param logits: The attention logits, of shape (..., num_queries, num_keys).
    :param factors: The cutoff factors of the keys. Broadcastable to the logits.
    :param established_power: Power of the cutoff factor with which a key counts as
        an established competitor for the attention received by the other keys.
    :return: The attention weights, normalized over the keys.
    """
    weights = torch.softmax(logits.masked_fill(factors <= 0.0, float("-inf")), dim=-1)

    # Established weight of the keys other than each one. Subtracting each term from
    # the total would be inaccurate for a key that dominates it, so we start from the
    # sum of all the terms but the largest.
    counted = weights * factors**established_power
    largest, index = counted.max(dim=-1, keepdim=True)
    rest = counted.scatter(-1, index, 0.0).sum(dim=-1, keepdim=True)
    # the clamp only acts when there are no other keys, or when a key outweighs all
    # the others by more than twelve orders of magnitude
    established = (rest + (largest - counted)).clamp(min=1e-12)

    weights = (
        factors * weights * established / (established + (1.0 - factors) * weights)
    )
    return weights / weights.sum(dim=-1, keepdim=True)
