from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .utilities import NeverRun, cutoff_func


class AttentionBlock(nn.Module):
    def __init__(self, total_dim, num_heads, dropout=0.0, epsilon=1e-15):
        super(AttentionBlock, self).__init__()

        self.input_linear = nn.Linear(total_dim, 3 * total_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(total_dim, total_dim)

        nn.init.xavier_uniform_(self.input_linear.weight)
        nn.init.constant_(self.input_linear.bias, 0.0)
        nn.init.constant_(self.output_linear.bias, 0.0)

        self.num_heads = num_heads
        self.epsilon = epsilon

        if total_dim % num_heads != 0:
            raise ValueError("total dimension is not divisible by the number of heads")
        self.head_dim = total_dim // num_heads
        self.preconditioning = 1.0 / np.sqrt(self.head_dim)

    def forward(self, x, multipliers: Optional[torch.Tensor] = None):
        initial_shape = x.shape
        x = self.input_linear(x)
        x = x.reshape(
            initial_shape[0], initial_shape[1], 3, self.num_heads, self.head_dim
        )
        x = x.permute(2, 0, 3, 1, 4)

        queries, keys, values = x[0], x[1], x[2]
        alpha = torch.matmul(queries, keys.transpose(-2, -1)) * self.preconditioning
        alpha = F.softmax(alpha, dim=-1)
        alpha = self.dropout(alpha)

        if multipliers is not None:
            alpha = alpha * multipliers[:, None, :, :]
            alpha = alpha / (alpha.sum(dim=-1)[..., None] + self.epsilon)

        x = torch.matmul(alpha, values).transpose(1, 2).reshape(initial_shape)
        x = self.output_linear(x)
        return x


class TransformerLayer(torch.nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dim_feedforward=512,
        dropout=0.0,
        activation=F.silu,
        transformer_type="PostLN",
    ):
        super(TransformerLayer, self).__init__()
        self.attention = AttentionBlock(d_model, n_heads, dropout=dropout)

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

    def forward(self, x, multipliers: Optional[torch.Tensor] = None):
        if self.transformer_type == "PostLN":
            x = self.norm_attention(x + self.dropout(self.attention(x, multipliers)))
            x = self.norm_mlp(x + self.mlp(x))
        if self.transformer_type == "PreLN":
            x = x + self.dropout(self.attention(self.norm_attention(x), multipliers))
            x = x + self.mlp(self.norm_mlp(x))
        return x


class Transformer(torch.nn.Module):
    def __init__(
        self,
        d_model,
        num_layers,
        n_heads,
        dim_feedforward=512,
        dropout=0.0,
        activation=F.silu,
        transformer_type="PostLN",
    ):
        super(Transformer, self).__init__()
        self.transformer_type = transformer_type

        self.final_norm = NeverRun()  # for torchscript
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

    def forward(self, x: torch.Tensor, multipliers: Optional[torch.Tensor] = None):
        for layer in self.layers:
            x = layer(x, multipliers)
        if self.transformer_type == "PreLN":
            x = self.final_norm(x)
        return x


class CartesianTransformer(torch.nn.Module):
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
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=torch.nn.SiLU(),
            transformer_type="PostLN",
        )

        input_dim = 4  # x, y, z, r

        self.r_embedding = nn.Linear(input_dim, d_model)

        if not is_first:
            n_merge = 3
        else:
            n_merge = 2

        self.compress = nn.Sequential(
            nn.Linear(n_merge * d_model, d_model),
            torch.nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.neighbor_embedder = NeverRun()  # for torchscript
        if not is_first:
            self.neighbor_embedder = nn.Embedding(n_atomic_species + 1, d_model)

        self.central_embedder = NeverRun()  # for torchscript
        self.central_scalar_embedding = NeverRun()  # for torchscript
        self.central_compress = NeverRun()  # for torchscript

        self.central_embedder = nn.Embedding(n_atomic_species + 1, d_model)

    def forward(
        self,
        batch_dict: Dict[str, torch.Tensor],
    ):
        x = batch_dict["x"]
        mask = batch_dict["mask"]
        central_species = batch_dict["central_species"]
        neighbor_species = batch_dict["neighbor_species"]
        input_messages = batch_dict["input_messages"]
        neighbor_lengths = torch.sqrt(torch.sum(x**2, dim=2) + 1e-15)[:, :, None]
        if not self.is_first:
            neighbor_embedding = self.neighbor_embedder(neighbor_species)
        else:
            neighbor_embedding = torch.empty(
                0, device=x.device, dtype=x.dtype
            )  # for torch script

        initial_n_tokens = x.shape[1]
        max_number = input_messages.shape[1]
        coordinates = [x, neighbor_lengths]
        coordinates = torch.cat(coordinates, dim=2)
        coordinates = self.r_embedding(coordinates)

        if not self.is_first:
            tokens = torch.cat([coordinates, neighbor_embedding, input_messages], dim=2)
        else:
            tokens = torch.cat([coordinates, input_messages], dim=2)

        tokens = self.compress(tokens)
        central_specie_embedding = self.central_embedder(central_species)
        central_token = central_specie_embedding

        tokens = torch.cat([central_token[:, None, :], tokens], dim=1)

        submask = torch.zeros(mask.shape[0], dtype=torch.bool, device=mask.device)
        total_mask = torch.cat([submask[:, None], mask], dim=1)

        lengths = torch.sqrt(torch.sum(x * x, dim=2) + 1e-16)
        with torch.profiler.record_function("cutoff_func"):
            multipliers = cutoff_func(lengths, self.cutoff, self.cutoff_width)
        sub_multipliers = torch.ones(mask.shape[0], device=mask.device)
        multipliers = torch.cat([sub_multipliers[:, None], multipliers], dim=1)
        multipliers[total_mask] = 0.0

        multipliers = multipliers[:, None, :]
        multipliers = multipliers.repeat(1, multipliers.shape[2], 1)

        output_messages = self.trans(
            tokens[:, : (max_number + 1), :],
            multipliers=multipliers[:, : (max_number + 1), : (max_number + 1)],
        )
        if max_number < initial_n_tokens:
            padding = torch.zeros(
                output_messages.shape[0],
                initial_n_tokens - max_number,
                output_messages.shape[2],
                device=output_messages.device,
            )
            output_messages = torch.cat([output_messages, padding], dim=1)

        return {
            "output_messages": output_messages[:, 1:, :],
            "central_token": output_messages[:, 0, :],
        }
