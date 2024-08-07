from typing import List

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


class Linear(torch.nn.Module):

    def __init__(self, n_feat_in, n_feat_out):
        super().__init__()
        self.linear_layer = torch.nn.Linear(n_feat_in, n_feat_out, bias=False)
        self.linear_layer.weight.data.normal_(0.0, 1.0)
        self.n_feat_in = n_feat_in

    def forward(self, x):
        return self.linear_layer(x) * self.n_feat_in**(-0.5)


class InvariantLinear(torch.nn.Module):

    def __init__(self, n_inputs: int) -> None:
        super().__init__()
        self.linear = Linear(n_inputs, 1)

    def forward(self, features: TensorMap) -> TensorMap:

        # assume invariant
        block = features.block({"o3_lambda": 0, "o3_sigma": 1})

        output_values = self.linear(block.values).squeeze(1)
        new_block = TensorBlock(
            values=output_values,
            samples=block.samples,
            components=[],
            properties=Labels(
                names=["energy"],
                values=torch.zeros(
                    (1, 1), dtype=torch.int32, device=block.values.device
                ),
            ),
        )

        return TensorMap(
            keys=Labels(
                names=["_"],
                values=torch.zeros(
                    1, 1, dtype=torch.int32, device=new_block.values.device
                ),
            ),
            blocks=[new_block],
        )


class InvariantMLP(torch.nn.Module):

    def __init__(self, n_inputs: int) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            Linear(n_inputs, 4*n_inputs),
            torch.nn.SiLU(),
            # Linear(4*n_inputs, 4*n_inputs),
            # torch.nn.SiLU(),
            # Linear(4*n_inputs, 4*n_inputs),
            # torch.nn.SiLU(),
            Linear(4*n_inputs, n_inputs),
        )

    def forward(self, features: TensorMap) -> TensorMap:

        # assume invariant
        block = features.block({"o3_lambda": 0, "o3_sigma": 1})

        output_values = self.mlp(block.values)
        new_block = TensorBlock(
            values=output_values,
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )

        return TensorMap(
            keys=Labels(
                names=["o3_lambda", "o3_sigma"],
                values=torch.tensor(
                    [[0, 1]], dtype=torch.int32, device=new_block.values.device
                ),
            ),
            blocks=[new_block],
        )


class EquivariantLinear(torch.nn.Module):

    def __init__(self, irreps, k_max_l, double=False) -> None:
        # double can be used to double the input dimension (used in tensor_sum.py)
        super().__init__()

        self.irreps = irreps

        # Register linear layers:
        self.linear_contractions = torch.nn.ModuleDict(
            {
                f"{L}_{S}": torch.nn.Sequential(
                    Linear((2*k_max_l[L] if double else k_max_l[L]), k_max_l[L]),
                )
                for L, S in irreps
            }
        )

    def forward(self, features: TensorMap) -> TensorMap:
        new_blocks: List[TensorBlock] = []

        for irrep_name, contraction in self.linear_contractions.items():
            split_irrep = irrep_name.split("_")
            L = int(split_irrep[0])
            S = int(split_irrep[1])
            block = features.block({"o3_lambda": L, "o3_sigma": S})
            new_values = contraction(block.values)
            new_block = TensorBlock(
                values=new_values,
                samples=block.samples,
                components=block.components,
                properties=Labels(
                    names=block.properties.names,
                    values=torch.arange(new_values.shape[-1], device=new_values.device).reshape(-1, 1),
                ),
            )
            new_blocks.append(new_block)

        # keys should always be in the correct order (because self.irreps is)
        return TensorMap(keys=features.keys, blocks=new_blocks)
