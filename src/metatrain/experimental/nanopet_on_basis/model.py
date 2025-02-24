from typing import Dict, List, Tuple, Union, Optional

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

from .utils import symmetrize_samples, keys_triu_center_type
from metatensor.torch.learn import ModuleMap


class NanoPetOnBasis(torch.torch.nn.Module):
    """
    Makes node (single-center) and edge (two-center) predictions on a spherical basis.
    """

    def __init__(
        self,
        atomic_types: List[int],
        in_keys_node: Optional[Labels] = None,
        out_properties_node: Optional[List[Labels]] = None,
        in_keys_edge: Optional[Labels] = None,
        out_properties_edge: Optional[List[Labels]] = None,
        pet_hypers=None,
        head_hidden_layer_widths=[64, 64, 64],
        standardizers: Dict[str, TensorMap] = None,
    ) -> None:
        """
        Some notes on current implementation:

        ``atomic_types`` is the complete list of atomic numbers that the model (may)
        ever see. Attempting to predict on an atomic configuration containing atomic
        types not present in this list will throw.

        However, ``in_keys_*`` can contains a subset of these atomic types. The
        underlying NanoPET will (for now) predict on all atoms present in the input
        system, but the head modules will transform these internal features to make
        predictions only on blocks specified in ``in_keys_*``.
        """

        super().__init__()

        # Extract node target metadata
        self.atomic_types = atomic_types

        if in_keys_node is not None:
            self.in_keys_node = in_keys_node
            self.out_properties_node = out_properties_node
            self.predict_nodes = True
        else:
            self.in_keys_node = None
            self.out_properties_node = None
            self.predict_nodes = False

        # Extract edge target metadata
        if in_keys_edge is not None:
            # Triangularize the edge keys and keep the corresponding properties
            self.in_keys_edge, self.out_properties_edge = keys_triu_center_type(
                in_keys_edge, out_properties_edge
            )
            self.predict_edges = True
        else:
            self.in_keys_edge = None
            self.out_properties_edge = None
            self.predict_edges = False

        # Instantiate NanoPET model
        if pet_hypers is None:
            pet_hypers = get_default_hypers("experimental.nanopet")["model"]
        self.nanopet = NanoPET(
            pet_hypers,
            DatasetInfo(
                length_unit="angstrom",
                atomic_types=self.atomic_types,  # NanoPET predicts on the global set
                # of atomic types
                targets={},
            ),
        )

        # Build node heads
        if self.predict_nodes:
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
        self._set_standardizers(standardizers)

    def _set_standardizers(self, standardizers):
        """
        Set the standardizers for the node and edge targets.
        Turns off gradients for the standardizer values.
        """
        if standardizers is None:
            self.standardizers = None
            return

        for name, tensor in standardizers.items():
            assert name in ["node_mean", "node_std", "edge_mean", "edge_std"]
            for block in tensor:
                block.values.requires_grad = False

        self.standardizers = standardizers

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
        if self.predict_nodes:
            predictions_node = symmetrize_predictions_node(
                self.atomic_types,
                pet_features["node"],
                self.in_keys_node,
                systems=systems,
            )
            predictions_node = self.node_heads(predictions_node)
            predictions_node = self._reshape_predictions(predictions_node, "node")

            if system_id is not None:
                predictions_node = reindex_tensormap(predictions_node, system_id)
        else:
            predictions_node = None

        # Next handle edges, if applicable
        if self.predict_edges:
            predictions_edge = symmetrize_predictions_edge(
                self.atomic_types,
                pet_features["edge"],
                self.in_keys_edge,
                systems=systems,
            )
            predictions_edge = self.edge_heads(predictions_edge)
            predictions_edge = self._reshape_predictions(predictions_edge, "edge")
            if system_id is not None:
                predictions_edge = reindex_tensormap(predictions_edge, system_id)

            predictions_edge = mts.sort(predictions_edge, "samples")
        else:
            predictions_edge = None

        # Add back the mean and revert the standardization, if applicable
        predictions_node, predictions_edge = self._add_mean_revert_std(
            predictions_node, predictions_edge
        )
        return predictions_node, predictions_edge

    def _add_mean_revert_std(
        self,
        predictions_node: TensorMap,
        predictions_edge: Optional[TensorMap] = None,
    ) -> TensorMap:
        """
        Adds back in the mean to invariant blocks, and reverts the standardization of
        all blocks.
        """
        if self.standardizers is None:
            return predictions_node, predictions_edge

        # TODO: does order matter here?
        if predictions_node is not None:
            if "node_mean" in self.standardizers:
                predictions_node = add_back_invariant_mean(
                    predictions_node,
                    self.standardizers["node_mean"],
                )
            if "node_std" in self.standardizers:
                predictions_node = revert_standardization(
                    predictions_node,
                    self.standardizers["node_std"],
                )
        if predictions_edge is not None:
            if "edge_mean" in self.standardizers:
                predictions_edge = add_back_invariant_mean(
                    predictions_edge,
                    self.standardizers["edge_mean"],
                )
            if "edge_std" in self.standardizers:
                predictions_edge = revert_standardization(
                    predictions_edge,
                    self.standardizers["edge_std"],
                )

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
        self.norm = torch.nn.LayerNorm(input_dim, device=device)
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
    atomic_types: List[int],
    predictions_node: TensorMap,
    in_keys_node: Labels,
    systems,
) -> TensorMap:
    """Symmetrize PET node predictions."""

    # Create a dictionary storing the atomic indices for each center type
    slice_nodes = {center_type: [] for center_type in atomic_types}
    for A, system in enumerate(systems):
        for i, center_type in enumerate(system.types):

            slice_nodes[int(center_type)].append([A, i])

    # Slice the predictions TensorMap to create blocks for the different center types
    # with the correct atomic samples
    node_blocks = []
    for key in in_keys_node:

        center_type = int(key["center_type"])

        block = mts.slice(
            predictions_node,
            "samples",
            Labels(
                ["system", "atom"],
                torch.tensor(slice_nodes[center_type], dtype=torch.int32).reshape(
                    -1, 2
                ),
            ),
        )[0]

        node_blocks.append(block)

    return TensorMap(in_keys_node, node_blocks)


def symmetrize_predictions_edge(
    atomic_types: List[int],
    predictions_edge: TensorMap,
    in_keys_edge: Labels,
    systems,
) -> TensorMap:
    """
    Symmetrize PET edge predictions
    """
    apply_permutational_symmetry = "block_type" in in_keys_edge.names

    slice_edges = {
        (first_atom_type, second_atom_type): []
        for first_atom_type in atomic_types
        for second_atom_type in atomic_types
    }
    for A, system in enumerate(systems):
        for i, first_atom_type in enumerate(system.types):
            for j, second_atom_type in enumerate(system.types):

                slice_edges[(int(first_atom_type), int(second_atom_type))].append(
                    [A, i, j]
                )

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
                torch.tensor(slice_edges[(Z1, Z2)], dtype=torch.int32).reshape(-1, 3),
            ),
        )[0]

        # Symmetrize
        if Z1 == Z2 and apply_permutational_symmetry:
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


def add_back_invariant_mean(tensor: TensorMap, mean_tensor: TensorMap) -> TensorMap:
    """
    Adds back the mean to the invariant blocks of the input ``tensor`` using the
    ``mean_tensor`` layer.
    """
    for key, mean_block in mean_tensor.items():
        if key not in tensor.keys:
            continue
        tensor_block = tensor[key]
        tensor_block.values[:] += mean_block.values

    return tensor


def revert_standardization(tensor: TensorMap, standardizer: TensorMap) -> TensorMap:
    """
    Un-standardizes the input ``tensor`` using the ``standardizer`` layer.
    """
    for key, block in tensor.items():
        block.values[:] *= standardizer.block(key).values

    return tensor
