from typing import List, Tuple, Union, Optional

import torch

import metatensor.torch as mts
from metatensor.torch.atomistic import ModelOutput
from metatensor.torch import Labels, TensorBlock, TensorMap


from metatrain.experimental.nanopet import NanoPET
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from elearn import symmetrize_samples, keys_triu_center_type
from metatensor.torch.learn import ModuleMap


class NanoPetOnBasis(torch.torch.nn.Module):
    """
    Makes node (single-center) and edge (two-center) predictions on a spherical basis.
    """

    def __init__(
        self,
        in_keys_node: Labels,
        out_properties_node: List[Labels],
        in_keys_edge: Labels = None,
        out_properties_edge: List[Labels] = None,
        pet_hypers=None,
        head_hidden_layer_widths=[64, 64, 64],
        standardizer_node: Optional[TensorMap] = None,
        standardizer_edge: Optional[TensorMap] = None,
    ) -> None:

        super().__init__()

        # Extract node target metadata
        self.in_keys_node = in_keys_node
        self.out_properties_node = out_properties_node
        self.atom_types = torch.unique(self.in_keys_node.column("center_type"))

        # Extract edge target metadata
        if in_keys_edge is not None:
            # Triangularize the edge keys and keep the corresponding properties
            self.in_keys_edge, self.out_properties_edge = keys_triu_center_type(
                in_keys_edge, out_properties_edge
            )
            self.in_keys_edge = in_keys_edge
            self.out_properties_edge = out_properties_edge
            self.predict_edges = True
        else:
            self.predict_edges = False

        # Instantiate NanoPET model
        if pet_hypers is None:
            pet_hypers = get_default_hypers("experimental.nanopet")["model"]
        # TODO: separate PET for each block!
        # self.nanopet = torch.nn.ModuleList(
        #     [
        #         NanoPET(
        #             pet_hypers,
        #             DatasetInfo(
        #                 length_unit="angstrom",
        #                 atomic_types=self.atom_types,
        #                 targets={},
        #             ),
        #         )
        #         for _ in self.in_keys_node
        #     ]
        # )
        self.nanopet = NanoPET(
            pet_hypers,
            DatasetInfo(
                length_unit="angstrom",
                atomic_types=self.atom_types,
                targets={},
            ),
        )

        # Build node heads
        self.node_heads = self._instantiate_heads(
            self.in_keys_node,
            self.out_properties_node,
            head_hidden_layer_widths,
        )

        # Build edge heads
        if self.predict_edges:
            self.edge_heads = self._instantiate_heads(
                self.in_keys_edge,
                self.out_properties_edge,
                head_hidden_layer_widths,
            )

        # Set the prediction (un)standardizer
        self._set_standardizers(standardizer_node, standardizer_edge)

    def _set_standardizers(self, standardizer_node, standardizer_edge):
        """
        Set the standardizers for the node and edge targets.
        Turns off gradients for the standardizer values.
        """
        # Node standardizer
        if standardizer_node is not None:
            for block in standardizer_node:
                block.values.requires_grad = False
                self.standardizer_node = standardizer_node
        else:
            self.standardizer_node = None

        # Edge standardizer
        if standardizer_edge is not None:
            for block in standardizer_edge:
                block.values.requires_grad = False
            self.standardizer_edge = standardizer_edge
        else:
            self.standardizer_edge = None

    def forward(
        self, systems, system_id: List[int] = None
    ) -> Tuple[TensorMap, Union[TensorMap, None]]:
        """
        Predicts the node and edge (if applicable) targets on a basis.
        """

        # Get the neighbor lists required for PET
        systems = [
            get_system_with_neighbor_lists(
                sys, get_requested_neighbor_lists(self.nanopet)
            )
            for sys in systems
        ]

        # Compute PET features. NOTE: currently this results in weight sharing between
        # different blocks for the PET part of the architecture.
        pet_features = self.nanopet(
            systems,
            {"features": ModelOutput(per_atom=True)},
        )["features"]

        # Symmetrize PET predictions and pass through head modules. First handle nodes.
        predictions_node = symmetrize_predictions_node(
            pet_features["node"],
            self.in_keys_node,
            systems=systems,
        )
        predictions_node = self.node_heads(predictions_node)
        predictions_node = self._reshape_predictions(predictions_node, "node")

        if system_id is not None:
            predictions_node = reindex_tensormap(predictions_node, system_id)

        # Next handle edges, if applicable
        if self.predict_edges:
            predictions_edge = symmetrize_predictions_edge(
                pet_features["edge"],
                self.in_keys_edge,
                systems=systems,
            )
            predictions_edge = self.edge_heads(predictions_edge)
            predictions_edge = self._reshape_predictions(predictions_edge, "edge")
            if system_id is not None:
                predictions_edge = reindex_tensormap(predictions_edge, system_id)

            if self.standardizer_node is not None:
                predictions_node = unstandardize_tensor(
                    predictions_node, self.standardizer_node
                )
            if self.standardizer_edge is not None:
                predictions_edge = unstandardize_tensor(
                    predictions_edge, self.standardizer_edge
                )

            return predictions_node, predictions_edge

        if self.standardizer_node is not None:
            predictions_node = unstandardize_tensor(
                predictions_node, self.standardizer_node
            )
        return predictions_node

    def _instantiate_heads(
        self,
        in_keys: Labels,
        out_properties: List[Labels],
        hidden_layer_widths: List[int],
    ) -> ModuleMap:
        """
        Builds a ModuleMap of MLPs for each node or edge block.
        """
        return ModuleMap(
            in_keys,
            [
                MLPModel(
                    in_features=self.nanopet.hypers["d_pet"],
                    out_features=len(out_props) * (2 * key["o3_lambda"] + 1),
                    hidden_layer_widths=hidden_layer_widths,
                )
                for key, out_props in zip(in_keys, out_properties)
            ],
        )

    def _reshape_predictions(
        self,
        predicted_features: TensorMap,
        feature_type: str = "node",
    ) -> TensorMap:
        """
        Reshapes the 2D blocks in `predicted_features` to 3D blocks with an equivariant
        component axis
        """
        # Get the target properties metadata
        if feature_type == "node":
            out_properties = self.out_properties_node
            in_keys = self.in_keys_node
        else:
            assert feature_type == "edge"
            out_properties = self.out_properties_edge
            in_keys = self.in_keys_edge

        # Reshape each block in turn
        predicted_blocks = []
        for key, out_props in zip(in_keys, out_properties):

            if key not in predicted_features.keys:
                continue
            predicted_block = predicted_features[key]
            reshaped_block = TensorBlock(
                values=predicted_block.values.reshape(
                    len(predicted_block.samples),
                    2 * key["o3_lambda"] + 1,  # new mu components axis
                    len(out_props),
                ),
                samples=predicted_block.samples,
                components=[
                    Labels(
                        ["o3_mu"],
                        torch.arange(
                            -key["o3_lambda"],
                            key["o3_lambda"] + 1,
                            dtype=torch.int64,
                        ).reshape(-1, 1),
                    ),
                ],
                properties=out_props,
            )
            predicted_blocks.append(reshaped_block)

        return TensorMap(predicted_features.keys, predicted_blocks)


# ===== HEAD MODULES ===== #


class ResidualBlock(torch.nn.Module):
    """
    ResidualBlock is a neural network module that implements a residual block with
    normalization, linear transformation, activation, and dropout.
    """

    def __init__(self, input_dim, output_dim, device=None):
        super().__init__()
        # self.norm = torch.nn.LayerNorm(input_dim, device=device)
        self.linear = torch.nn.Linear(input_dim, output_dim, device=device)
        self.activation = torch.nn.SiLU()
        # self.dropout = torch.nn.Dropout(0.5)

        if input_dim != output_dim:
            self.projection = torch.nn.Linear(input_dim, output_dim, device=device)
        else:
            self.projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        # out = self.norm(x)
        out = x
        out = self.linear(out)
        out = self.activation(out)
        # out = self.dropout(out)

        if self.projection is not None:
            residual = self.projection(residual)
        return out + residual


class MLPModel(torch.nn.Module):
    """
    MLPModel is a multi-layer perceptron model with residual blocks.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layer_widths: List[int],
        device="cpu",
    ) -> None:

        super().__init__()

        # Initialize the layers
        layers = [ResidualBlock(in_features, hidden_layer_widths[0], device=device)]
        for layer_i, hidden_layer_width in enumerate(hidden_layer_widths[1:], start=1):
            layers.append(
                ResidualBlock(
                    hidden_layer_widths[layer_i - 1], hidden_layer_width, device=device
                )
            )
        layers.append(
            torch.nn.Linear(hidden_layer_widths[-1], out_features, device=device)
        )

        # Build the sequential
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ===== HELPER FUNCTIONS ===== #


def symmetrize_predictions_node(
    predictions_node: TensorMap,
    in_keys_node: Labels,
    systems,
) -> TensorMap:
    """
    Symmetrize PET node predictions
    """

    slice_nodes = {}
    for key in in_keys_node:
        Z = int(key["center_type"])
        if Z not in slice_nodes:
            slice_nodes[Z] = []
    for A, system in enumerate(systems):
        for i, Z in enumerate(system.types):
            Z = int(Z)
            slice_nodes[Z].append([A, i])

    # Nodes
    node_blocks = []
    for key in in_keys_node:

        Z = int(key["center_type"])

        block = mts.slice(
            predictions_node,
            "samples",
            Labels(
                ["system", "atom"],
                torch.tensor(slice_nodes[Z], dtype=torch.int32).reshape(-1, 2),
            ),
        )[0]

        node_blocks.append(block)

    return TensorMap(in_keys_node, node_blocks)


def symmetrize_predictions_edge(
    predictions_edge: TensorMap,
    in_keys_edge: Labels,
    systems,
) -> TensorMap:
    """
    Symmetrize PET edge predictions
    """

    slice_edges = {}
    for key in in_keys_edge:
        Z1 = int(key["first_atom_type"])
        Z2 = int(key["second_atom_type"])
        if (Z1, Z2) not in slice_edges:
            slice_edges[(Z1, Z2)] = []
    for A, system in enumerate(systems):
        for i, Z1 in enumerate(system.types):
            Z1 = int(Z1)
            for j, Z2 in enumerate(system.types):
                if Z1 > Z2:
                    continue
                Z2 = int(Z2)
                slice_edges[(Z1, Z2)].append([A, i, j])

    # Edges (properly symmetrized)
    edge_blocks = []
    for key in in_keys_edge:

        Z1 = int(key["first_atom_type"])
        Z2 = int(key["second_atom_type"])

        # Slice to the relevant types, which could leave a block with zero samples
        block = mts.slice(
            predictions_edge,
            "samples",
            Labels(
                ["system", "first_atom", "second_atom"],
                torch.tensor(slice_edges[Z1, Z2], dtype=torch.int32).reshape(-1, 3),
            ),
        )[0]

        # Symmetrize
        if Z1 == Z2:
            block_plus, block_minus = symmetrize_samples(block)
            if key["block_type"] == 1:
                edge_blocks.append(block_plus)
            elif key["block_type"] == -1:
                edge_blocks.append(block_minus)
            else:
                raise ValueError(f"Block type must be 1 or -1 for Z1=Z2={Z1}")
        else:
            edge_blocks.append(block)

    return TensorMap(in_keys_edge, edge_blocks)


def reindex_tensormap(
    tensor: TensorMap,
    system_ids: List[int],
) -> TensorMap:
    """
    Takes a single TensorMap `tensor` containing data on multiple systems and re-indexes
    the "system" dimension of the samples. Assumes input has numeric system indices from
    {0, ..., N_system - 1} (inclusive), and maps these indices one-to-one with those
    passed in ``system_ids``.
    """
    assert tensor.sample_names[0] == "system"

    index_mapping = {i: A for i, A in enumerate(system_ids)}

    def new_row(row):
        return [index_mapping[row[0].item()]] + [i for i in row[1:]]

    new_blocks = []
    for block in tensor.blocks():
        new_samples = mts.Labels(
            names=block.samples.names,
            values=torch.tensor(
                [new_row(row) for row in block.samples.values],
                dtype=torch.int32,
            ).reshape(-1, len(block.samples.names)),
        )
        new_block = mts.TensorBlock(
            values=block.values,
            samples=new_samples,
            components=block.components,
            properties=block.properties,
        )
        new_blocks.append(new_block)

    return mts.TensorMap(tensor.keys, new_blocks)


def unstandardize_tensor(tensor: TensorMap, standardizer: TensorMap) -> TensorMap:
    """
    Standardizes the input ``tensor`` using the ``standardizer`` layer.
    """
    for key, block in tensor.items():
        block.values[:] *= standardizer.block(key).values

    return tensor
