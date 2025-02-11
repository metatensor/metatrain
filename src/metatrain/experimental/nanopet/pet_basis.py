from typing import List

import torch

from metatensor.torch.atomistic import ModelOutput
from metatensor.torch import Labels, TensorMap
import metatensor.torch as mts

from metatrain.experimental.nanopet import NanoPET
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from .elearn import symmetrize_samples
from metatensor.torch.learn import ModuleMap


class ResidualBlock(torch.nn.Module):
    """
    ResidualBlock is a neural network module that implements a residual block with
    normalization, linear transformation, activation, and dropout.

    Parameters
    ----------
    input_dim : int
        The dimension of the input features.
    output_dim : int
        The dimension of the output features.
    device : torch.device, optional
        The device on which to place the tensors (default is None).

    Attributes
    ----------
    norm : torch.nn.LayerNorm
        Layer normalization applied to the input.
    linear : torch.nn.Linear
        Linear transformation applied to the normalized input.
    activation : torch.nn.SiLU
        Activation function applied to the linear transformation.
    dropout : torch.nn.Dropout
        Dropout applied to the activated output.
    projection : torch.nn.Linear or None
        Linear transformation applied to the residual connection if
        input_dim != output_dim, otherwise None.

    Methods
    -------
    forward(x)
        Forward pass of the residual block.
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

    def forward(self, x):
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

    Parameters
    ----------
    input_dim : int
        The dimension of the input features.
    output_dim : int
        The dimension of the output features.
    hidden_layers : int
        The number of hidden layers in the model.
    neurons_per_layer : int
        The number of neurons per hidden layer.
    device : str, optional
        The device on which to place the tensors (default is "cpu").

    Attributes
    ----------
    model : torch.nn.Sequential
        The sequential container of the model layers.

    Methods
    -------
    forward(x)
        Forward pass of the MLP model.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layers,
        neurons_per_layer,
        device="cpu",
    ):

        super(MLPModel, self).__init__()

        layers = [ResidualBlock(input_dim, neurons_per_layer, device=device)]
        for _ in range(hidden_layers - 1):
            layers.append(
                ResidualBlock(neurons_per_layer, neurons_per_layer, device=device)
            )
        layers.append(torch.nn.Linear(neurons_per_layer, output_dim, device=device))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def symmetrize_pet_features(
    pet_features,
    node_targets,
    edge_targets,
    slice_nodes=None,
    slice_edges=None,
    systems=None,
):
    """
    Symmetrize PET features for nodes and edges.

    Parameters
    ----------
    pet_features : dict
        Dictionary containing node and edge features.
    node_targets : TensorMap
        TensorMap containing node target information.
    edge_targets : TensorMap
        TensorMap containing edge target information.
    slice_nodes : dict, optional
        Dictionary containing slices of nodes (default is None).
    slice_edges : dict, optional
        Dictionary containing slices of edges (default is None).
    systems : list, optional
        List of systems to process (default is None).

    Returns
    -------
    node_features_tensor : TensorMap
        TensorMap containing symmetrized node features.
    edge_features_tensor : TensorMap
        TensorMap containing symmetrized edge features.
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

    # Nodes
    node_features = pet_features["node"]
    node_keys = []
    node_blocks = []
    for Z in slice_nodes:
        block = mts.slice(
            node_features,
            "samples",
            Labels(["system", "atom"], torch.tensor(slice_nodes[Z])),
        )[0]

        for key_value in node_targets.keys.values[
            torch.where(
                torch.all(
                    node_targets.keys.view(["center_type"]).values == torch.tensor([Z]),
                    dim=1,
                )
            )
        ]:

            node_keys.append(key_value)
            node_blocks.append(block)

    node_features_tensor = TensorMap(
        Labels(["o3_lambda", "o3_sigma", "center_type"], torch.stack(node_keys)),
        node_blocks,
    )

    # Edges (properly symmetrized)
    edge_features = pet_features["edge"]
    edge_keys = []
    edge_blocks = []
    for Z1, Z2 in slice_edges:
        block = mts.slice(
            edge_features,
            "samples",
            Labels(
                ["system", "first_atom", "second_atom"],
                torch.tensor(slice_edges[Z1, Z2]),
            ),
        )[0]

        # Symmetrize
        if Z1 == Z2:
            block_plus, block_minus = symmetrize_samples(block)

            for key_value in edge_targets.keys.values[
                torch.where(
                    torch.all(
                        edge_targets.keys.view(
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
            for key_value in edge_targets.keys.values[
                torch.where(
                    torch.all(
                        edge_targets.keys.view(
                            ["first_atom_type", "second_atom_type"]
                        ).values
                        == torch.tensor([Z1, Z2]),
                        dim=1,
                    )
                )
            ]:
                edge_keys.append(key_value)
                edge_blocks.append(block)

    edge_features_tensor = TensorMap(
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

    return node_features_tensor, edge_features_tensor


class NanoPETOnBasis(torch.torch.nn.Module):
    """
    NanoPETPerPair is a neural network module that processes node and edge features
    using a NanoPET model and MLP models for each feature type.

    Parameters
    ----------
    node_metadata : TensorMap
        TensorMap containing metadata for node features.
    edge_metadata : TensorMap
        TensorMap containing metadata for edge features.
    hidden_layers : int, optional
        The number of hidden layers in the MLP models (default is 3).
    neurons_per_layer : int, optional
        The number of neurons per hidden layer in the MLP models (default is 256).

    Attributes
    ----------
    nanopet : NanoPET
        The NanoPET model used to extract features.
    node_metadata : TensorMap
        TensorMap containing metadata for node features.
    edge_metadata : TensorMap
        TensorMap containing metadata for edge features.
    node_heads : ModuleMap
        ModuleMap containing MLP models for each node feature type.
    edge_heads : ModuleMap
        ModuleMap containing MLP models for each edge feature type.

    Methods
    -------
    forward(systems)
        Forward pass of the NanoPETPerPair model.
    """

    def __init__(
        self,
        node_metadata,
        edge_metadata=None,
        pet_hypers=None,
        hidden_layers=3,
        neurons_per_layer=64,
    ):

        super().__init__()

        if pet_hypers is None:
            pet_hypers = get_default_hypers("experimental.nanopet")["model"]

        self.atom_types = torch.unique(node_metadata.keys.column("center_type"))

        self.nanopet = NanoPET(
            pet_hypers,
            DatasetInfo(
                length_unit="angstrom",
                atomic_types=self.atom_types,
                targets={},
            ),
        )

        # Instantiate node heads
        self.node_metadata = node_metadata
        self.in_keys_node = node_metadata.keys
        self.out_properties_edge = [b.properties for b in node_metadata.blocks()]

        self.node_heads = self._instantiate_heads(
            self.in_keys_node,
            self.out_properties_edge,
            hidden_layers,
            neurons_per_layer,
        )

        if edge_metadata is not None:
            # Instantiate edge heads
            indices = []
            for i, k in enumerate(edge_metadata.keys):
                if k["first_atom_type"] > k["second_atom_type"]:
                    indices.append(i)
            self.edge_metadata = mts.drop_blocks(
                edge_metadata,
                mts.Labels(
                    edge_metadata.keys.names,
                    edge_metadata.keys.values[indices],
                ),
            )
            self.in_keys_edge = edge_metadata.keys
            self.out_properties_edge = [b.properties for b in edge_metadata.blocks()]

            self.edge_heads = self._instantiate_heads(
                self.in_keys_edge,
                self.out_properties_edge,
                hidden_layers,
                neurons_per_layer,
            )

    def _instantiate_heads(
        self,
        in_keys: Labels,
        out_properties: List[int],
        hidden_layers: int,
        neurons_per_layer: int,
    ) -> ModuleMap:
        return ModuleMap(
            in_keys,
            [
                MLPModel(
                    input_dim=self.nanopet.hypers["d_pet"],
                    output_dim=len(out_props) * (2 * key["o3_lambda"] + 1),
                    hidden_layers=hidden_layers,
                    neurons_per_layer=neurons_per_layer,
                )
                for key, out_props in zip(in_keys, out_properties)
            ],
        )

    def _reshape_predictions(self, predicted_features, target_properties):
        prediction = []
        for (key, block), out_props in zip(
            predicted_features.items(), target_properties
        ):
            o3_lambda = key["o3_lambda"]
            prediction.append(
                mts.TensorBlock(
                    values=block.values.reshape(
                        len(block.samples),
                        2 * o3_lambda + 1,
                        len(out_props),
                    ),
                    samples=block.samples,
                    components=[
                        mts.Labels(
                            ["o3_mu"],
                            torch.arange(
                                -o3_lambda, o3_lambda + 1, dtype=torch.int64
                            ).reshape(-1, 1),
                        ),
                    ],
                    properties=out_props,
                )
            )
        return TensorMap(predicted_features.keys, prediction)

    def forward(self, systems):

        systems = [
            get_system_with_neighbor_lists(
                sys, get_requested_neighbor_lists(self.nanopet)
            )
            for sys in systems
        ]

        pet_features = self.nanopet(
            systems,
            {"features": ModelOutput(per_atom=True)},
        )["features"]

        node_features, edge_features = symmetrize_pet_features(
            pet_features, self.node_metadata, self.edge_metadata, systems=systems
        )

        node_predictions = self.node_heads(node_features)
        edge_predictions = self.edge_heads(edge_features)

        nodes = self._reshape_predictions(node_predictions, self.node_properties)
        edges = self._reshape_predictions(edge_predictions, self.edge_properties)

        return nodes, edges
