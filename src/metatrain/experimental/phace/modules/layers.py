from typing import Dict, List

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


class Linear(torch.nn.Module):

    def __init__(self, n_feat_in, n_feat_out):
        super().__init__()
        self.linear_layer = torch.nn.Linear(n_feat_in, n_feat_out, bias=False)
        self.linear_layer.weight.data.normal_(0.0, 1.0)
        self.n_feat_in = n_feat_in

    def forward(self, x):
        return self.linear_layer(x) * self.n_feat_in ** (-0.5)


class InvariantMLP(torch.nn.Module):

    def __init__(self, n_inputs: int) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            Linear(n_inputs, 4 * n_inputs),
            torch.nn.SiLU(),
            # Linear(4*n_inputs, 4*n_inputs),
            # torch.nn.SiLU(),
            # Linear(4*n_inputs, 4*n_inputs),
            # torch.nn.SiLU(),
            Linear(4 * n_inputs, n_inputs),
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
                    Linear((2 * k_max_l[L] if double else k_max_l[L]), k_max_l[L]),
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
                    values=torch.arange(
                        new_values.shape[-1], device=new_values.device
                    ).reshape(-1, 1),
                ),
            )
            new_blocks.append(new_block)

        # keys should always be in the correct order (because self.irreps is)
        return TensorMap(keys=features.keys, blocks=new_blocks)


class EquivariantLastLayer(torch.nn.Module):

    output_components: Dict[str, List[Labels]]
    output_properties: Dict[str, Labels]

    def __init__(
        self,
        irreps,
        k_max_l,
        output_components: List[List[Labels]],
        output_properties: List[Labels],
    ) -> None:
        super().__init__()

        # Register linear layers:
        self.linear_contractions = torch.nn.ModuleDict(
            {
                f"{L}_{S}": torch.nn.Sequential(
                    Linear(k_max_l[L], len(properties)),
                )
                for (L, S), properties in zip(irreps, output_properties)
            }
        )

        self.output_components = {
            f"{L}_{S}": components
            for (L, S), components in zip(irreps, output_components)
        }
        self.output_properties = {
            f"{L}_{S}": properties
            for (L, S), properties in zip(irreps, output_properties)
        }
        self.single_label = Labels.single()
        self.keys = Labels(
            names=["o3_lambda", "o3_sigma"],
            values=torch.tensor([[L, S] for L, S in irreps], dtype=torch.int32),
        )

    def forward(self, features: TensorMap) -> TensorMap:
        # move components and properties to device if necessary
        device = features.device
        for irrep in self.output_properties:
            self.output_properties[irrep] = self.output_properties[irrep].to(device)
        for irrep in self.output_components:
            self.output_components[irrep] = [
                component.to(device) for component in self.output_components[irrep]
            ]
        self.keys = self.keys.to(device)
        self.single_label = self.single_label.to(device)

        new_blocks: List[TensorBlock] = []

        for irrep_name, contraction in self.linear_contractions.items():
            split_irrep = irrep_name.split("_")
            L = int(split_irrep[0])
            S = int(split_irrep[1])
            block = features.block({"o3_lambda": L, "o3_sigma": S})
            new_values = contraction(block.values)
            if len(self.output_components[irrep_name]) == 0:
                new_values = new_values.squeeze(1)  # remove component dimension
            new_block = TensorBlock(
                values=new_values,
                samples=block.samples,
                components=self.output_components[irrep_name],
                properties=self.output_properties[irrep_name],
            )
            new_blocks.append(new_block)

        if len(new_blocks) == 1 and len(new_blocks[0].components) == 0:
            # no components, "scalar" convention
            new_keys = self.single_label
        else:
            new_keys = self.keys

        # keys should always be in the correct order (because self.irreps is)
        return TensorMap(keys=new_keys, blocks=new_blocks)


class NothingLayer(torch.nn.Module):
    # useful when the head for an output is a simple linear layer

    def forward(self, features: TensorMap) -> TensorMap:
        return features
