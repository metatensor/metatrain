from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import ModelOutput
from metatensor.torch.learn import ModuleMap

from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    System,
)
from ...utils.dtype import dtype_to_str
from ...utils.metadata import append_metadata_references

from ..nanopet import NanoPET
from metatrain.utils.data import DatasetInfo
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)
from .utils import (
    symmetrize_predictions_node,
    symmetrize_predictions_edge,
    add_back_invariant_mean,
    revert_standardization,
)


class NanoPETBasis(torch.torch.nn.Module):
    """
    Makes node (single-center) and edge (two-center) predictions on a spherical basis.
    """

    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float64, torch.float32]
    __default_metadata__ = ModelMetadata(
        references={"architecture": ["https://arxiv.org/abs/2305.19302v3"]}
    )

    def __init__(
        self,
        model_hypers: Dict,
        dataset_info: DatasetInfo,
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

        # Store hypers
        self.hypers = model_hypers
        self.dataset_info = dataset_info
        self.new_outputs = list(dataset_info.targets.keys())
        self.atomic_types = dataset_info.atomic_types
        self.outputs = {}  # TODO!

        # Initialize base NanoPET
        self.nanopet = NanoPET(
            self.hypers["nanopet"],
            DatasetInfo(
                length_unit=self.dataset_info.length_unit,
                atomic_types=self.atomic_types,
                targets={},
            ),
        )

        # Extract node target metadata
        if self.dataset_info.targets.get("node") is not None:
            self.in_keys_node = self.dataset_info.targets["node"].layout.keys
            self.out_properties_node = [
                self.dataset_info.targets["node"].layout[key].properties for key in self.in_keys_node
            ]
            self.predict_nodes = True
        else:
            self.in_keys_node = None
            self.out_properties_node = None
            self.predict_nodes = False

        # Extract edge target metadata
        if self.dataset_info.targets.get("edge") is not None:
            self.in_keys_edge = self.dataset_info.targets["edge"].layout.keys
            self.out_properties_edge = [
                self.dataset_info.targets["edge"].layout[key].properties for key in self.in_keys_edge
            ]
            self.predict_edges = True
        else:
            self.in_keys_edge = None
            self.out_properties_edge = None
            self.predict_edges = False

        # Build node heads
        if self.predict_nodes:
            self.node_heads = self._instantiate_heads(
                self.in_keys_node,
                self.out_properties_node,
                self.hypers["basis"]["head_hidden_layer_widths"],
            )

        # Build edge heads
        if self.predict_edges:
            self.edge_heads = self._instantiate_heads(
                self.in_keys_edge,
                self.out_properties_edge,
                self.hypers["basis"]["head_hidden_layer_widths"],
            )

        # Set the prediction (un)standardizer
        self._set_standardizers(getattr(self.dataset_info, "standardizers", None))

    def _set_standardizers(self, standardizers) -> None:
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

    def restart(self, dataset_info: DatasetInfo) -> "NanoPETBasis":
        # merge old and new dataset info
        merged_info = self.dataset_info.union(dataset_info)
        new_atomic_types = [
            at for at in merged_info.atomic_types if at not in self.atomic_types
        ]
        new_targets = {
            key: value
            for key, value in merged_info.targets.items()
            if key not in self.dataset_info.targets
        }
        self.has_new_targets = len(new_targets) > 0

        if len(new_atomic_types) > 0:
            raise ValueError(
                f"New atomic types found in the dataset: {new_atomic_types}. "
                "The nanoPET model does not support adding new atomic types."
            )

        # register new outputs as new last layers
        for target_name, target in new_targets.items():
            self._add_output(target_name, target)

        self.dataset_info = merged_info

        return self

    def forward(
        self, systems: List[System],
        outputs: Dict[str, ModelOutput],  # TODO!
        selected_atoms: Optional[Labels],  # TODO!
    ) -> Dict[str, TensorMap]:
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
            # TODO: remove sorting
            predictions_edge = mts.sort(predictions_edge, "samples")
        else:
            predictions_edge = None

        # Add back the mean and revert the standardization, if applicable
        predictions_node, predictions_edge = self._add_mean_revert_std(
            predictions_node, predictions_edge
        )
        return {"node": predictions_node, "edge": predictions_edge}
    
    @classmethod
    def load_checkpoint(cls, path: Union[str, Path]) -> "NanoPETBasis":
        # Load the checkpoint
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        model_data = checkpoint["model_data"]
        model_state_dict = checkpoint["model_state_dict"]

        # Create the model
        model = cls(**model_data)
        state_dict_iter = iter(model_state_dict.values())
        next(state_dict_iter)  # skip `species_to_species_index` buffer (int)
        dtype = next(state_dict_iter).dtype
        model.to(dtype).load_state_dict(model_state_dict)

        return model
    
    def export(
        self, metadata: Optional[ModelMetadata] = None
    ) -> MetatensorAtomisticModel:
        dtype = next(self.parameters()).dtype
        if dtype not in self.__supported_dtypes__:
            raise ValueError(f"unsupported dtype {dtype} for NanoPET")

        # Make sure the model is all in the same dtype
        # For example, after training, the additive models could still be in
        # float64
        self.to(dtype)

        # TODO: CompositionModel for NanoPETBasis?
        # # Additionally, the composition model contains some `TensorMap`s that cannot
        # # be registered correctly with Pytorch. This funciton moves them:
        # self.additive_models[0]._move_weights_to_device_and_dtype(
        #     torch.device("cpu"), torch.float64
        # )

        interaction_ranges = [self.hypers["nanopet"]["num_gnn_layers"] * self.hypers["nanopet"]["cutoff"]]
        # TODO: CompositionModel for NanoPETBasis?
        # for additive_model in self.additive_models:
        #     if hasattr(additive_model, "cutoff_radius"):
        #         interaction_ranges.append(additive_model.cutoff_radius)
        #     if self.long_range:
        #         interaction_ranges.append(torch.inf)
        interaction_range = max(interaction_ranges)

        capabilities = ModelCapabilities(
            outputs=self.outputs,
            atomic_types=self.atomic_types,
            interaction_range=interaction_range,
            length_unit=self.dataset_info.length_unit,
            supported_devices=self.__supported_devices__,
            dtype=dtype_to_str(dtype),
        )

        if metadata is None:
            metadata = ModelMetadata()

        append_metadata_references(metadata, self.__default_metadata__)

        return MetatensorAtomisticModel(self.eval(), metadata, capabilities)

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
