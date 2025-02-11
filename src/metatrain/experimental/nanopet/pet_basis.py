from typing import List, Tuple, Union

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

from elearn import symmetrize_samples
from metatensor.torch.learn import ModuleMap


class NanoPetOnBasis(torch.torch.nn.Module):
    """
    Makes node (single-center) and edge (two-center) predictions on a spherical basis.
    """

    def __init__(
        self,
        node_metadata,
        edge_metadata=None,
        pet_hypers=None,
        head_hidden_layer_widths=[64, 64, 64],
    ) -> None:

        super().__init__()

        # Extract node target metadata
        self.in_keys_node = node_metadata.keys
        self.out_properties_node = [node_metadata[key].properties for key in self.in_keys_node]
        self.atom_types = torch.unique(self.in_keys_node.column("center_type"))
        
        # Extract edge target metadata
        if edge_metadata is not None:
            self.in_keys_edge = edge_metadata.keys
            self.out_properties_edge = [edge_metadata[key].properties for key in self.in_keys_edge]
            self.predict_edges = True
        else:
            self.predict_edges = False
        

        # TODO: should this belong here?
        # self.in_keys_edge, self.out_properties_edge = elearn.keys_triu_center_type(self.in_keys_edge, self.out_properties_edge)

        # Instantiate NanoPET model
        if pet_hypers is None:
            pet_hypers = get_default_hypers("experimental.nanopet")["model"]
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

    def forward(self, systems) -> Tuple[TensorMap, Union[TensorMap, None]]:
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

        # Next handle edges, if applicable
        if self.predict_edges:
            predictions_edge = symmetrize_predictions_edge(
                pet_features["edge"],
                self.in_keys_node,
                systems=systems,
            )
            predictions_edge = self.edge_heads(predictions_edge)
            predictions_edge = self._reshape_predictions(predictions_edge, "edge")
        else:
            predictions_edge = None

        return predictions_node, predictions_edge
    
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
        else:
            assert feature_type == "edge"
            out_properties = self.out_properties_edge

        # Reshape each block in turn
        predicted_blocks = []
        for key, out_props in zip(
            predicted_features.keys, out_properties
        ):
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
                            -key["o3_lambda"], key["o3_lambda"] + 1,
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
        self.norm = torch.nn.LayerNorm(input_dim, device=device)
        self.linear = torch.nn.Linear(input_dim, output_dim, device=device)
        self.activation = torch.nn.SiLU()
        self.dropout = torch.nn.Dropout(0.5)

        if input_dim != output_dim:
            self.projection = torch.nn.Linear(input_dim, output_dim, device=device)
        else:
            self.projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm(x)
        out = self.linear(out)
        out = self.activation(out)
        out = self.dropout(out)

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
                ResidualBlock(hidden_layer_widths[layer_i - 1], hidden_layer_width, device=device)
            )
        layers.append(torch.nn.Linear(hidden_layer_widths[-1], out_features, device=device))

        # Build the sequential
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    

# ===== HELPER FUNCTIONS ===== #

def symmetrize_predictions_node(
    predictions_node: TensorMap,
    in_keys_node: Labels,
    slice_nodes=None,
    systems=None,
) -> TensorMap:
    """
    Symmetrize PET node predictions
    """

    if slice_nodes is None:
        assert systems is not None
        slice_nodes = {}
        for A, system in enumerate(systems):
            for i, Z in enumerate(system.types):
                Z = int(Z)
                if Z not in slice_nodes:
                    slice_nodes[Z] = []
                slice_nodes[Z].append([A, i])

    # Nodes
    node_keys = []
    node_blocks = []
    for Z in slice_nodes:
        block = mts.slice(
            predictions_node,
            "samples",
            Labels(["system", "atom"], torch.tensor(slice_nodes[Z])),
        )[0]

        for key_value in in_keys_node.values[
            torch.where(
                torch.all(
                    in_keys_node.view(["center_type"]).values == torch.tensor([Z]),
                    dim=1,
                )
            )
        ]:

            node_keys.append(key_value)
            node_blocks.append(block)


    return TensorMap(
        Labels(
            ["o3_lambda", "o3_sigma", "center_type"],
            torch.stack(node_keys)
        ),
        node_blocks,
    )

def symmetrize_predictions_edge(
    predictions_edge: dict,
    in_keys_edge: Labels = None,
    slice_edges=None,
    systems=None,
) -> TensorMap:
    """
    Symmetrize PET edge predictions
    """
    
    if slice_edges is None:
        assert systems is not None
        slice_edges = {}
        for A, system in enumerate(systems):
            for i, Z1 in enumerate(system.types):
                Z1 = int(Z1)
                for j, Z2 in enumerate(system.types):
                    Z2 = int(Z2)
                    if (Z1, Z2) not in slice_edges:
                        slice_edges[(Z1, Z2)] = []
                    slice_edges[(Z1, Z2)].append([A, i, j])

    # Edges (properly symmetrized)
    edge_keys = []
    edge_blocks = []
    for Z1, Z2 in slice_edges:
        block = mts.slice(
            predictions_edge,
            "samples",
            Labels(
                ["system", "first_atom", "second_atom"],
                torch.tensor(slice_edges[Z1, Z2]),
            ),
        )[0]

        # Symmetrize
        if Z1 == Z2:
            block_plus, block_minus = symmetrize_samples(block)

            for key_value in in_keys_edge.values[
                torch.where(
                    torch.all(
                        in_keys_edge.view(
                            ["first_atom_type", "second_atom_type"]
                        ).values
                        == torch.tensor([Z1, Z2]),
                        dim=1,
                    )
                )
            ]:
                edge_keys.append(key_value)
                if key_value[-1] == 1:
                    edge_blocks.append(block_plus)
                elif key_value[-1] == -1:
                    edge_blocks.append(block_minus)
                else:
                    raise ValueError("Block type must be 1 or -1 for Z1=Z2={Z1}")
        else:
            for key_value in in_keys_edge.values[
                torch.where(
                    torch.all(
                        in_keys_edge.view(
                            ["first_atom_type", "second_atom_type"]
                        ).values
                        == torch.tensor([Z1, Z2]),
                        dim=1,
                    )
                )
            ]:
                edge_keys.append(key_value)
                edge_blocks.append(block)

    return TensorMap(
        Labels(
            [
                "o3_lambda",
                "o3_sigma",
                "first_atom_type",
                "second_atom_type",
                "block_type",
            ],
            torch.stack(edge_keys),
        ),
        edge_blocks,
    )