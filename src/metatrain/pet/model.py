import logging
import warnings
from math import prod
from typing import Any, Dict, List, Literal, Optional, Tuple

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.operations._add import _add_block_block
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)

from metatrain.utils.abc import ModelInterface
from metatrain.utils.additive import ZBL, CompositionModel
from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.dtype import dtype_to_str
from metatrain.utils.long_range import DummyLongRangeFeaturizer, LongRangeFeaturizer
from metatrain.utils.metadata import merge_metadata
from metatrain.utils.scaler import Scaler
from metatrain.utils.sum_over_atoms import sum_over_atoms

from . import checkpoints
from .modules.finetuning import apply_finetuning_strategy
from .modules.structures import systems_to_batch
from .modules.transformer import CartesianTransformer
from .modules.utilities import cutoff_func


AVAILABLE_FEATURIZERS = ["feedforward", "residual"]


class PET(ModelInterface):
    """
    Metatrain-native implementation of the PET architecture.

    Originally proposed in work (https://arxiv.org/abs/2305.19302v3),
    and published in the `pet` package (https://github.com/spozdn/pet).

    :param hypers: Hyperparameters for the PET model. See the documentation for details.
    :param dataset_info: Information about the dataset, including atomic types and
        targets.
    """

    __checkpoint_version__ = 7
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float32, torch.float64]
    __default_metadata__ = ModelMetadata(
        references={"architecture": ["https://arxiv.org/abs/2305.19302v3"]}
    )
    component_labels: Dict[str, List[List[Labels]]]
    NUM_FEATURE_TYPES: int = 2  # node + edge features

    def __init__(self, hypers: Dict, dataset_info: DatasetInfo) -> None:
        super().__init__(hypers, dataset_info, self.__default_metadata__)

        # Cache frequently accessed hyperparameters
        self.cutoff = float(self.hypers["cutoff"])
        self.cutoff_width = float(self.hypers["cutoff_width"])
        self.d_pet = self.hypers["d_pet"]
        self.d_node = self.hypers["d_node"]
        self.d_head = self.hypers["d_head"]
        self.d_feedforward = self.hypers["d_feedforward"]
        self.num_heads = self.hypers["num_heads"]
        self.num_gnn_layers = self.hypers["num_gnn_layers"]
        self.num_attention_layers = self.hypers["num_attention_layers"]
        self.normalization = self.hypers["normalization"]
        self.activation = self.hypers["activation"]
        self.transformer_type = self.hypers["transformer_type"]
        self.featurizer_type = self.hypers["featurizer_type"]

        self.atomic_types = dataset_info.atomic_types
        self.requested_nl = NeighborListOptions(
            cutoff=self.cutoff,
            full_list=True,
            strict=True,
        )
        num_atomic_species = len(self.atomic_types)
        self.gnn_layers = torch.nn.ModuleList(
            [
                CartesianTransformer(
                    self.hypers,
                    self.d_pet,
                    self.num_heads,
                    self.d_node,
                    self.d_feedforward,
                    self.num_attention_layers,
                    self.normalization,
                    self.activation,
                    self.transformer_type,
                    num_atomic_species,
                    layer_index == 0,  # is first layer
                )
                for layer_index in range(self.num_gnn_layers)
            ]
        )
        if self.featurizer_type not in AVAILABLE_FEATURIZERS:
            raise ValueError(
                f"Unknown featurizer type: {self.featurizer_type}. "
                f"Available options are: {AVAILABLE_FEATURIZERS}"
            )
        if self.featurizer_type == "feedforward":
            self.num_readout_layers = 1
            self.combination_norms = torch.nn.ModuleList(
                [torch.nn.LayerNorm(2 * self.d_pet) for _ in range(self.num_gnn_layers)]
            )
            self.combination_mlps = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.Linear(2 * self.d_pet, 2 * self.d_pet),
                        torch.nn.SiLU(),
                        torch.nn.Linear(2 * self.d_pet, self.d_pet),
                    )
                    for _ in range(self.num_gnn_layers)
                ]
            )
        else:
            self.num_readout_layers = self.num_gnn_layers
            self.combination_norms = torch.nn.ModuleList()
            self.combination_mlps = torch.nn.ModuleList()

        self.node_embedders = torch.nn.ModuleList(
            [
                torch.nn.Embedding(num_atomic_species + 1, self.d_node)
                for _ in range(self.num_readout_layers)
            ]
        )
        self.edge_embedder = torch.nn.Embedding(num_atomic_species + 1, self.d_pet)

        self.node_heads = torch.nn.ModuleDict()
        self.edge_heads = torch.nn.ModuleDict()
        self.node_last_layers = torch.nn.ModuleDict()
        self.edge_last_layers = torch.nn.ModuleDict()
        self.last_layer_feature_size = (
            self.num_readout_layers * self.d_head * self.NUM_FEATURE_TYPES
        )

        self.outputs = {
            "features": ModelOutput(unit="", per_atom=True)
        }  # the model is always capable of outputting the internal features

        self.output_shapes: Dict[str, Dict[str, List[int]]] = {}
        self.key_labels: Dict[str, Labels] = {}
        self.property_labels: Dict[str, List[Labels]] = {}
        self.component_labels: Dict[str, List[List[Labels]]] = {}
        self.target_names: List[str] = []
        for target_name, target_info in dataset_info.targets.items():
            self.target_names.append(target_name)
            self._add_output(target_name, target_info)

        self.register_buffer(
            "species_to_species_index",
            torch.full((max(self.atomic_types) + 1,), -1),
        )
        for i, species in enumerate(self.atomic_types):
            self.species_to_species_index[species] = i

        # long-range module
        if self.hypers["long_range"]["enable"]:
            self.long_range = True
            if not self.hypers["long_range"]["use_ewald"]:
                warnings.warn(
                    "Training PET with the LongRangeFeaturizer initialized "
                    "with `use_ewald=False` causes instabilities during training. "
                    "The `use_ewald` variable will be force-switched to `True`. "
                    "during training.",
                    UserWarning,
                    stacklevel=2,
                )
            self.long_range_featurizer = LongRangeFeaturizer(
                hypers=self.hypers["long_range"],
                feature_dim=self.d_pet,
                neighbor_list_options=self.requested_nl,
            )
        else:
            self.long_range = False
            self.long_range_featurizer = DummyLongRangeFeaturizer()  # for torchscript

        # additive models: these are handled by the trainer at training
        # time, and they are added to the output at evaluation time
        composition_model = CompositionModel(
            hypers={},
            dataset_info=DatasetInfo(
                length_unit=dataset_info.length_unit,
                atomic_types=self.atomic_types,
                targets={
                    target_name: target_info
                    for target_name, target_info in dataset_info.targets.items()
                    if CompositionModel.is_valid_target(target_name, target_info)
                },
            ),
        )
        additive_models = [composition_model]

        # Adds the ZBL repulsion model if requested
        if self.hypers["zbl"]:
            additive_models.append(
                ZBL(
                    {},
                    dataset_info=DatasetInfo(
                        length_unit=dataset_info.length_unit,
                        atomic_types=self.atomic_types,
                        targets={
                            target_name: target_info
                            for target_name, target_info in dataset_info.targets.items()
                            if ZBL.is_valid_target(target_name, target_info)
                        },
                    ),
                )
            )
        self.additive_models = torch.nn.ModuleList(additive_models)

        # scaler: this is also handled by the trainer at training time
        self.scaler = Scaler(hypers={}, dataset_info=dataset_info)

        self.single_label = Labels.single()

        self.finetune_config: Dict[str, Any] = {}

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs

    def restart(self, dataset_info: DatasetInfo) -> "PET":
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
                "The PET model does not support adding new atomic types."
            )

        # register new outputs as new last layers
        for target_name, target in new_targets.items():
            self.target_names.append(target_name)
            self._add_output(target_name, target)

        self.dataset_info = merged_info

        # restart the composition and scaler models
        self.additive_models[0].restart(
            dataset_info=DatasetInfo(
                length_unit=dataset_info.length_unit,
                atomic_types=self.atomic_types,
                targets={
                    target_name: target_info
                    for target_name, target_info in dataset_info.targets.items()
                    if CompositionModel.is_valid_target(target_name, target_info)
                },
            ),
        )
        self.scaler.restart(dataset_info)
        return self

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [self.requested_nl]

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        """
        Forward pass of the PET model.

        The forward pass processes atomic systems through multiple stages to produce
        predictions for the requested outputs. The computation follows a graph neural
        network architecture with attention-based message passing.

        **Stage 0: Input Preparation**

        The input systems are first converted into a batched representation containing:

        - `element_indices_nodes` [n_atoms]: Atomic species of the central atoms
        - `element_indices_neighbors` [n_edges]: Atomic species of neighboring atoms
        - `edge_vectors` [n_atoms, max_num_neighbors, 3]: Cartesian edge vectors
          between central atoms and their neighbors
        - `padding_mask` [n_atoms, max_num_neighbors]: Mask indicating real vs padded
          neighbors
        - `neighbors_index` [n_atoms, max_num_neighbors]: Indices of neighboring atoms
          for each central atom
        - `reversed_neighbor_list` [n_atoms, max_num_neighbors]: For each center atom
          `i` and its neighbor `j`, the position of atom `i` in the neighbor list of
          atom `j`
        - `system_indices` [n_atoms]: System index for each central atom
        - `sample_labels` [n_atoms, 2]: Metatensor Labels containing indices of each
          atom in each system

        **Stage 1: Feature Computation via GNN Layers**

        Node and edge representations are computed by iterating through the GNN layers
        following one of two featurization strategies:

        - **Feedforward featurization**: Features are propagated through all
          `num_gnn_layers` GNN layers sequentially, using only the final layer outputs
          for readout. At each layer, forward and reversed edge messages are combined
          using combination MLPs to enable bidirectional information flow.

        - **Residual featurization**: Intermediate node and edge features from each
          GNN layer are saved and used during readout. Edge messages between layers are
          averaged to maintain information from all hops.

        During this stage, the model:

        - Embeds atomic species into learned node and edge representations
        - Applies Cartesian transformer layers to update features via attention
        - Uses reversed neighbor lists to enable bidirectional message passing, where
          the new input message from atom `j` to atom `i` in GNN layer N+1 is the
          reversed message from atom `i` to atom `j` in GNN layer N
        - Applies cutoff functions to weight interactions by distance

        If the long-range module is enabled, electrostatic features computed via Ewald
        summation (during training) or Particle-Particle Particle Mesh Ewald (P3M)
        (during evaluation) are added to the node features from each GNN layer.

        **Stage 2: Intermediate Feature Output (Optional)**

        If "features" is requested in the outputs, node and edge features from all
        layers are concatenated to produce intermediate representations. Edge features
        are summed over neighbors with cutoff weighting to obtain per-node
        contributions. This output can be used for transfer learning or analysis.

        **Stage 3: Last Layer Feature Computation**

        For each requested output, output-specific heads (shallow MLPs with two linear
        layers and SiLU activations) are applied to both node and edge features from
        each GNN layer. This produces last layer features that are specialized for each
        prediction target. These features can be optionally returned as auxiliary
        outputs (e.g., "mtt::aux::energy_last_layer_features") for analysis or
        transfer learning.

        **Stage 4: Atomic Predictions**

        Final linear layers are applied to the last layer features to produce per-atom
        predictions for each requested output:

        - Node and edge last layer features are processed through separate linear
          layers for each output block
        - Contributions from all GNN layers are summed
        - Edge contributions are summed over neighbors with cutoff weighting
        - For rank-2 Cartesian tensors (e.g., stress), predictions are symmetrized and
          normalized by cell volume
        - Multiple tensor blocks per output are handled independently

        **Post-processing (Evaluation Only)**

        During evaluation (not training), the following transformations are applied:

        1. **Scaling**: Predictions are scaled using learned or configured scale
           factors
        2. **Additive contributions**: Composition model and optional ZBL repulsion
           contributions are added to the predictions

        :param systems: List of `metatomic.torch.System` objects to process. Each
            system should contain atomic positions, species, and cell information, with
            neighbor lists computed according to `requested_neighbor_lists()`.
        :param outputs: Dictionary of requested outputs in the format
            {output_name: ModelOutput(...)}. The model supports:

            - Target properties (energy, forces, stress, etc.)
            - "features": intermediate representations from Stage 2
            - Auxiliary last layer features (e.g.,
              "mtt::aux::energy_last_layer_features")

        :param selected_atoms: Optional `metatensor.torch.Labels` object specifying a
            subset of atoms for which to compute outputs. If `None`, all atoms are
            included. This is useful for computing properties for specific atomic
            environments.
        :return: Dictionary of `metatensor.torch.TensorMap` objects containing the
            requested outputs. Each TensorMap contains per-atom or per-structure
            predictions (depending on the ModelOutput configuration) with appropriate
            metatensor metadata (samples, components, properties).
        """
        device = systems[0].device
        return_dict: Dict[str, TensorMap] = {}
        nl_options = self.requested_neighbor_lists()[0]

        if self.single_label.values.device != device:
            self._move_labels_to_device(device)

        # **Stage 0: Input Preparation**

        (
            element_indices_nodes,
            element_indices_neighbors,
            edge_vectors,
            padding_mask,
            neighbors_index,
            reversed_neighbor_list,
            system_indices,
            sample_labels,
        ) = systems_to_batch(
            systems,
            nl_options,
            self.atomic_types,
            self.species_to_species_index,
            selected_atoms,
        )

        # the scaled_dot_product_attention function from torch cannot do
        # double backward, so we will use manual attention if needed
        use_manual_attention = edge_vectors.requires_grad and self.training

        edge_distances = torch.sqrt(torch.sum(edge_vectors**2, dim=2) + 1e-15)
        cutoff_factors = cutoff_func(edge_distances, self.cutoff, self.cutoff_width)
        cutoff_factors[~padding_mask] = 0.0

        # **Stage 1: Feature Computation via GNN Layers**

        featurizer_inputs: Dict[str, torch.Tensor] = dict(
            element_indices_nodes=element_indices_nodes,
            element_indices_neighbors=element_indices_neighbors,
            edge_vectors=edge_vectors,
            neighbors_index=neighbors_index,
            reversed_neighbor_list=reversed_neighbor_list,
            padding_mask=padding_mask,
            edge_distances=edge_distances,
            cutoff_factors=cutoff_factors,
        )
        node_features_list, edge_features_list = self._calculate_features(
            featurizer_inputs,
            use_manual_attention=use_manual_attention,
        )

        # If the long-range module is activated, we add the long-range features
        # on top of the node features

        if self.long_range:
            long_range_features = self._calculate_long_range_features(
                systems, node_features_list, edge_distances, padding_mask
            )
            for i in range(self.num_readout_layers):
                node_features_list[i] = (
                    node_features_list[i] + long_range_features
                ) * 0.5**0.5

        # **Stage 2: Intermediate Feature Output (Optional)**

        if "features" in outputs:
            features_dict = self._get_output_features(
                node_features_list,
                edge_features_list,
                cutoff_factors,
                selected_atoms,
                sample_labels,
                outputs,
            )
            # Since return_dict.update(features_dict) is not Torch-Scriptable,
            # we use a simple iteration over the features_dict items.
            for k, v in features_dict.items():
                return_dict[k] = v

        # **Stage 3: Last Layer Feature Computation**

        node_last_layer_features_dict, edge_last_layer_features_dict = (
            self._calculate_last_layer_features(
                node_features_list,
                edge_features_list,
            )
        )

        last_layer_features_dict = self._get_output_last_layer_features(
            node_last_layer_features_dict,
            edge_last_layer_features_dict,
            cutoff_factors,
            selected_atoms,
            sample_labels,
            outputs,
        )

        for k, v in last_layer_features_dict.items():
            return_dict[k] = v

        # **Stage 4: Atomic Predictions**

        node_atomic_predictions_dict, edge_atomic_predictions_dict = (
            self._calculate_atomic_predictions(
                node_last_layer_features_dict,
                edge_last_layer_features_dict,
                padding_mask,
                cutoff_factors,
                outputs,
            )
        )

        atomic_predictions_dict = self._get_output_atomic_predictions(
            systems,
            node_atomic_predictions_dict,
            edge_atomic_predictions_dict,
            edge_vectors,
            system_indices,
            sample_labels,
            outputs,
            selected_atoms,
        )

        for k, v in atomic_predictions_dict.items():
            return_dict[k] = v

        # **Post-processing (Evaluation Only)**

        if not self.training:
            # at evaluation, we also introduce the scaler and additive contributions
            return_dict = self.scaler(return_dict)
            for additive_model in self.additive_models:
                outputs_for_additive_model: Dict[str, ModelOutput] = {}
                for name, output in outputs.items():
                    if name in additive_model.outputs:
                        outputs_for_additive_model[name] = output
                additive_contributions = additive_model(
                    systems,
                    outputs_for_additive_model,
                    selected_atoms,
                )
                for name in additive_contributions:
                    # # TODO: uncomment this after metatensor.torch.add is updated to
                    # # handle sparse sums
                    # return_dict[name] = metatensor.torch.add(
                    #     return_dict[name],
                    #     additive_contributions[name].to(
                    #         device=return_dict[name].device,
                    #         dtype=return_dict[name].dtype
                    #         ),
                    # )

                    # TODO: "manual" sparse sum: update to metatensor.torch.add after
                    # sparse sum is implemented in metatensor.operations
                    output_blocks: List[TensorBlock] = []
                    for k, b in return_dict[name].items():
                        if k in additive_contributions[name].keys:
                            output_blocks.append(
                                _add_block_block(
                                    b,
                                    additive_contributions[name]
                                    .block(k)
                                    .to(device=b.device, dtype=b.dtype),
                                )
                            )
                        else:
                            output_blocks.append(b)
                    return_dict[name] = TensorMap(return_dict[name].keys, output_blocks)

        return return_dict

    @torch.compile(
        mode="max-autotune",
        dynamic=True,
        fullgraph=True,
    )
    def _calculate_features(
        self, inputs: Dict[str, torch.Tensor], use_manual_attention: bool
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Calculate node and edge features using the selected featurization strategy.
        Returns lists of feature tensors from GNN layers.

        :param inputs: Dictionary containing input tensors required for feature
            computation
        :param use_manual_attention: Whether to use manual attention computation
            (required for double backward when edge vectors require gradients)
        :return: Tuple of two lists:
            - List of node feature tensors from each GNN layer
            - List of edge feature tensors from each GNN layer
        """
        if self.featurizer_type == "feedforward":
            return self._feedforward_featurization_impl(inputs, use_manual_attention)
        else:
            return self._residual_featurization_impl(inputs, use_manual_attention)

    def _feedforward_featurization_impl(
        self, inputs: Dict[str, torch.Tensor], use_manual_attention: bool
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Feedforward featurization: iterates features through all GNN layers,
        returning only the final layer outputs. Uses combination MLPs to mix
        forward and reversed edge messages at each layer.

        :param inputs: Dictionary containing input tensors required for feature
            computation
        :param use_manual_attention: Whether to use manual attention computation
            (required for double backward when edge vectors require gradients)
        :return: Tuple of two lists:
            - List of node feature tensors from the final GNN layer
            - List of edge feature tensors from the final GNN layer
        """
        node_features_list: List[torch.Tensor] = []
        edge_features_list: List[torch.Tensor] = []

        input_node_embeddings = self.node_embedders[0](inputs["element_indices_nodes"])
        input_edge_embeddings = self.edge_embedder(inputs["element_indices_neighbors"])
        for combination_norm, combination_mlp, gnn_layer in zip(
            self.combination_norms, self.combination_mlps, self.gnn_layers, strict=True
        ):
            output_node_embeddings, output_edge_embeddings = gnn_layer(
                input_node_embeddings,
                input_edge_embeddings,
                inputs["element_indices_neighbors"],
                inputs["edge_vectors"],
                inputs["padding_mask"],
                inputs["edge_distances"],
                inputs["cutoff_factors"],
                use_manual_attention,
            )

            # The GNN contraction happens by reordering the messages,
            # using a reversed neighbor list, so the new input message
            # from atom `j` to atom `i` in on the GNN layer N+1 is a
            # reversed message from atom `i` to atom `j` on the GNN layer N.
            input_node_embeddings = output_node_embeddings
            new_input_edge_embeddings = output_edge_embeddings[
                inputs["neighbors_index"], inputs["reversed_neighbor_list"]
            ]
            # input_messages = 0.5 * (output_edge_embeddings + new_input_messages)
            concatenated = torch.cat(
                [output_edge_embeddings, new_input_edge_embeddings], dim=-1
            )
            input_edge_embeddings = (
                input_edge_embeddings
                + output_edge_embeddings
                + combination_mlp(combination_norm(concatenated))
            )

        node_features_list.append(input_node_embeddings)
        edge_features_list.append(input_edge_embeddings)
        return node_features_list, edge_features_list

    def _residual_featurization_impl(
        self, inputs: Dict[str, torch.Tensor], use_manual_attention: bool
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Residual featurization: saves intermediate features from each GNN layer
        for use in readout. Averages forward and reversed edge messages between layers.

        :param inputs: Dictionary containing input tensors required for feature
            computation
        :param use_manual_attention: Whether to use manual attention computation
            (required for double backward when edge vectors require gradients)
        :return: Tuple of two lists:
            - List of node feature tensors from the final GNN layer
            - List of edge feature tensors from the final GNN layer
        """
        node_features_list: List[torch.Tensor] = []
        edge_features_list: List[torch.Tensor] = []
        input_edge_embeddings = self.edge_embedder(inputs["element_indices_neighbors"])
        for node_embedder, gnn_layer in zip(
            self.node_embedders, self.gnn_layers, strict=True
        ):
            input_node_embeddings = node_embedder(inputs["element_indices_nodes"])
            output_node_embeddings, output_edge_embeddings = gnn_layer(
                input_node_embeddings,
                input_edge_embeddings,
                inputs["element_indices_neighbors"],
                inputs["edge_vectors"],
                inputs["padding_mask"],
                inputs["edge_distances"],
                inputs["cutoff_factors"],
                use_manual_attention,
            )
            node_features_list.append(output_node_embeddings)
            edge_features_list.append(output_edge_embeddings)

            # The GNN contraction happens by reordering the messages,
            # using a reversed neighbor list, so the new input message
            # from atom `j` to atom `i` in on the GNN layer N+1 is a
            # reversed message from atom `i` to atom `j` on the GNN layer N.
            new_input_messages = output_edge_embeddings[
                inputs["neighbors_index"], inputs["reversed_neighbor_list"]
            ]
            input_edge_embeddings = 0.5 * (input_edge_embeddings + new_input_messages)
        return node_features_list, edge_features_list

    def _calculate_long_range_features(
        self,
        systems: List[System],
        node_features_list: List[torch.Tensor],
        edge_distances: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate long-range electrostatic features using Ewald summation.
        Forces use_ewald=True during training for stability.

        :param systems: List of `metatomic.torch.System` objects to process.
        :param node_features_list: List of node feature tensors from each GNN layer.
        :param edge_distances: Tensor of edge distances [n_atoms, max_num_neighbors].
        :param padding_mask: Boolean mask indicating real vs padded neighbors
            [n_atoms, max_num_neighbors].
        :return: Tensor of long-range features [n_atoms, d_pet].
        """
        if self.training:
            # Currently, the long-range implementation show instabilities
            # during training if P3MCalculator is used instead of the
            # EwaldCalculator. We will use the EwaldCalculator for training.
            self.long_range_featurizer.use_ewald = True
        flattened_lengths = edge_distances[padding_mask]
        short_range_features = (
            torch.stack(node_features_list).sum(dim=0)
            * (1 / len(node_features_list)) ** 0.5
        )
        long_range_features = self.long_range_featurizer(
            systems, short_range_features, flattened_lengths
        )
        return long_range_features

    def _get_output_features(
        self,
        node_features_list: List[torch.Tensor],
        edge_features_list: List[torch.Tensor],
        cutoff_factors: torch.Tensor,
        selected_atoms: Optional[Labels],
        sample_labels: Labels,
        requested_outputs: Dict[str, ModelOutput],
    ) -> Dict[str, TensorMap]:
        """
        Concatenate node and edge features from all layers into intermediate
        feature representations. Edge features are summed with cutoff weighting.

        :param node_features_list: List of node feature tensors from each GNN layer.
        :param edge_features_list: List of edge feature tensors from each GNN layer.
        :param cutoff_factors: Tensor of cutoff factors for edge distances
            [n_atoms, max_num_neighbors].
        :param selected_atoms: Optional Labels specifying a subset of atoms to include.
        :param sample_labels: Labels for all atoms in the batch [n_atoms, 2].
        :param requested_outputs: Dictionary of requested outputs.
        :return: Dictionary mapping "features" to a TensorMap of intermediate
            representations, either per-atom or summed over atoms.
        """
        features_dict: Dict[str, TensorMap] = {}
        node_features = torch.cat(node_features_list, dim=1)
        edge_features = torch.cat(edge_features_list, dim=2)
        edge_features = (edge_features * cutoff_factors[:, :, None]).sum(dim=1)
        features = torch.cat([node_features, edge_features], dim=1)

        feature_tmap = TensorMap(
            keys=self.single_label,
            blocks=[
                TensorBlock(
                    values=features,
                    samples=sample_labels,
                    components=[],
                    properties=Labels(
                        names=["feature"],
                        values=torch.arange(
                            features.shape[-1], device=features.device
                        ).reshape(-1, 1),
                        assume_unique=True,
                    ),
                )
            ],
        )
        if selected_atoms is not None:
            feature_tmap = mts.slice(
                feature_tmap,
                axis="samples",
                selection=selected_atoms,
            )
        if requested_outputs["features"].per_atom:
            features_dict["features"] = feature_tmap
        else:
            features_dict["features"] = sum_over_atoms(feature_tmap)
        return features_dict

    def _calculate_last_layer_features(
        self,
        node_features_list: List[torch.Tensor],
        edge_features_list: List[torch.Tensor],
    ) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[torch.Tensor]]]:
        """
        Apply output-specific heads to node and edge features from each GNN layer.
        Returns dictionaries mapping output names to lists of head-transformed features.

        :param node_features_list: List of node feature tensors from each GNN layer.
        :param edge_features_list: List of edge feature tensors from each GNN layer.
        :return: Tuple of two dictionaries:
            - Dictionary mapping output names to lists of node last layer features
            - Dictionary mapping output names to lists of edge last layer features
        """
        node_last_layer_features_dict: Dict[str, List[torch.Tensor]] = {}
        edge_last_layer_features_dict: Dict[str, List[torch.Tensor]] = {}

        # Calculating node last layer features
        for output_name, node_heads in self.node_heads.items():
            if output_name not in node_last_layer_features_dict:
                node_last_layer_features_dict[output_name] = []
            for i, node_head in enumerate(node_heads):
                node_last_layer_features_dict[output_name].append(
                    node_head(node_features_list[i])
                )

        # Calculating edge last layer features
        for output_name, edge_heads in self.edge_heads.items():
            if output_name not in edge_last_layer_features_dict:
                edge_last_layer_features_dict[output_name] = []
            for i, edge_head in enumerate(edge_heads):
                edge_last_layer_features_dict[output_name].append(
                    edge_head(edge_features_list[i])
                )

        return node_last_layer_features_dict, edge_last_layer_features_dict

    def _get_output_last_layer_features(
        self,
        node_last_layer_features_dict: Dict[str, List[torch.Tensor]],
        edge_last_layer_features_dict: Dict[str, List[torch.Tensor]],
        cutoff_factors: torch.Tensor,
        selected_atoms: Optional[Labels],
        sample_labels: Labels,
        requested_outputs: Dict[str, ModelOutput],
    ) -> Dict[str, TensorMap]:
        """
        Combine node and edge last layer features for requested last layer
        features output. Edge features are summed with cutoff weighting.

        :param node_last_layer_features_dict: Dictionary mapping output names to
            lists of node last layer features.
        :param edge_last_layer_features_dict: Dictionary mapping output names to
            lists of edge last layer features.
        :param cutoff_factors: Tensor of cutoff factors for edge distances
            [n_atoms, max_num_neighbors].
        :param selected_atoms: Optional Labels specifying a subset of atoms to include.
        :param sample_labels: Labels for all atoms in the batch [n_atoms, 2].
        :param requested_outputs: Dictionary of requested outputs.
        :return: Dictionary mapping requested last layer features output names
            to TensorMaps of last layer features, either per-atom or summed over atoms.
        """
        last_layer_features_dict: Dict[str, List[torch.Tensor]] = {}
        last_layer_features_outputs: Dict[str, TensorMap] = {}
        for output_name in node_last_layer_features_dict.keys():
            if not should_compute_last_layer_features(output_name, requested_outputs):
                continue
            if output_name not in last_layer_features_dict:
                last_layer_features_dict[output_name] = []
            for i in range(len(node_last_layer_features_dict[output_name])):
                node_last_layer_features = node_last_layer_features_dict[output_name][i]
                edge_last_layer_features = edge_last_layer_features_dict[output_name][i]
                edge_last_layer_features = (
                    edge_last_layer_features * cutoff_factors[:, :, None]
                ).sum(dim=1)
                last_layer_features_dict[output_name].append(node_last_layer_features)
                last_layer_features_dict[output_name].append(edge_last_layer_features)

        for output_name in requested_outputs:
            if not (
                output_name.startswith("mtt::aux::")
                and output_name.endswith("_last_layer_features")
            ):
                continue
            base_name = output_name.replace("mtt::aux::", "").replace(
                "_last_layer_features", ""
            )
            # the corresponding output could be base_name or mtt::base_name
            if f"mtt::{base_name}" in last_layer_features_dict:
                base_name = f"mtt::{base_name}"
            last_layer_features_values = torch.cat(
                last_layer_features_dict[base_name], dim=1
            )
            last_layer_feature_tmap = TensorMap(
                keys=self.single_label,
                blocks=[
                    TensorBlock(
                        values=last_layer_features_values,
                        samples=sample_labels,
                        components=[],
                        properties=Labels(
                            names=["feature"],
                            values=torch.arange(
                                last_layer_features_values.shape[-1],
                                device=last_layer_features_values.device,
                            ).reshape(-1, 1),
                            assume_unique=True,
                        ),
                    )
                ],
            )
            if selected_atoms is not None:
                last_layer_feature_tmap = mts.slice(
                    last_layer_feature_tmap,
                    axis="samples",
                    selection=selected_atoms,
                )
            last_layer_features_options = requested_outputs[output_name]
            if last_layer_features_options.per_atom:
                last_layer_features_outputs[output_name] = last_layer_feature_tmap
            else:
                last_layer_features_outputs[output_name] = sum_over_atoms(
                    last_layer_feature_tmap
                )
        return last_layer_features_outputs

    def _calculate_atomic_predictions(
        self,
        node_last_layer_features_dict: Dict[str, List[torch.Tensor]],
        edge_last_layer_features_dict: Dict[str, List[torch.Tensor]],
        padding_mask: torch.Tensor,
        cutoff_factors: torch.Tensor,
        outputs: Dict[str, ModelOutput],
    ) -> Tuple[
        Dict[str, List[List[torch.Tensor]]], Dict[str, List[List[torch.Tensor]]]
    ]:
        """
        Apply final linear layers to last layer features to produce
        per-atom predictions. Handles multiple blocks per output and sums
        edge contributions with cutoff weighting.

        :param node_last_layer_features_dict: Dictionary mapping output names to
            lists of node last layer features.
        :param edge_last_layer_features_dict: Dictionary mapping output names to
            lists of edge last layer features.
        :param padding_mask: Boolean mask indicating real vs padded neighbors
            [n_atoms, max_num_neighbors].
        :param cutoff_factors: Tensor of cutoff factors for edge distances
            [n_atoms, max_num_neighbors].
        :param outputs: Dictionary of requested outputs.
        :return: Tuple of two dictionaries:
            - Dictionary mapping output names to lists of lists of node atomic
              prediction tensors (one list per GNN layer, one tensor per block)
            - Dictionary mapping output names to lists of lists of edge atomic
              prediction tensors (one list per GNN layer, one tensor per block)
        """
        node_atomic_predictions_dict: Dict[str, List[List[torch.Tensor]]] = {}
        edge_atomic_predictions_dict: Dict[str, List[List[torch.Tensor]]] = {}

        # Computing node atomic predictions. Since we have last layer features
        # for each GNN layer, and each last layer can have multiple blocks,
        # we apply each last layer block to each of the last layer features.

        for output_name, node_last_layers in self.node_last_layers.items():
            if output_name in outputs:
                node_atomic_predictions_dict[output_name] = torch.jit.annotate(
                    List[List[torch.Tensor]], []
                )
                for i, node_last_layer in enumerate(node_last_layers):
                    node_last_layer_features = node_last_layer_features_dict[
                        output_name
                    ][i]
                    node_atomic_predictions_by_block: List[torch.Tensor] = []
                    for node_last_layer_by_block in node_last_layer.values():
                        node_atomic_predictions_by_block.append(
                            node_last_layer_by_block(node_last_layer_features)
                        )
                    node_atomic_predictions_dict[output_name].append(
                        node_atomic_predictions_by_block
                    )

        # Computing edge atomic predictions. Following the same logic as above,
        # we (1) iterate over the last layer features and last layer blocks, and (2)
        # sum the edge features with cutoff factors to get their per-node contribution.

        for output_name, edge_last_layers in self.edge_last_layers.items():
            if output_name in outputs:
                edge_atomic_predictions_dict[output_name] = torch.jit.annotate(
                    List[List[torch.Tensor]], []
                )
                for i, edge_last_layer in enumerate(edge_last_layers):
                    edge_last_layer_features = edge_last_layer_features_dict[
                        output_name
                    ][i]
                    edge_atomic_predictions_by_block: List[torch.Tensor] = []
                    for edge_last_layer_by_block in edge_last_layer.values():
                        edge_atomic_predictions = edge_last_layer_by_block(
                            edge_last_layer_features
                        )
                        expanded_padding_mask = padding_mask[..., None].repeat(
                            1, 1, edge_atomic_predictions.shape[2]
                        )
                        edge_atomic_predictions = torch.where(
                            ~expanded_padding_mask, 0.0, edge_atomic_predictions
                        )
                        edge_atomic_predictions_by_block.append(
                            (edge_atomic_predictions * cutoff_factors[:, :, None]).sum(
                                dim=1
                            )
                        )
                    edge_atomic_predictions_dict[output_name].append(
                        edge_atomic_predictions_by_block
                    )

        return node_atomic_predictions_dict, edge_atomic_predictions_dict

    def _get_output_atomic_predictions(
        self,
        systems: List[System],
        node_atomic_predictions_dict: Dict[str, List[List[torch.Tensor]]],
        edge_atomic_predictions_dict: Dict[str, List[List[torch.Tensor]]],
        edge_vectors: torch.Tensor,
        system_indices: torch.Tensor,
        sample_labels: Labels,
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        """
        Combine node and edge atomic predictions into final TensorMaps.
        Handles rank-2 Cartesian tensors by symmetrizing them.
        Returns per-atom or per-structure predictions based on output configuration.

        :param systems: List of `metatomic.torch.System` objects to process.
        :param node_atomic_predictions_dict: Dictionary mapping output names to
            lists of lists of node atomic prediction tensors (one list per GNN layer,
            one tensor per block).
        :param edge_atomic_predictions_dict: Dictionary mapping output names to
            lists of lists of edge atomic prediction tensors (one list per GNN layer,
            one tensor per block).
        :param edge_vectors: Tensor of edge vectors [n_atoms, max_num_neighbors, 3].
        :param system_indices: Tensor mapping each atom to its system index
            [n_atoms].
        :param sample_labels: Labels for all atoms in the batch [n_atoms, 2].
        :param outputs: Dictionary of requested outputs.
        :param selected_atoms: Optional Labels specifying a subset of atoms to include.
        :return: Dictionary mapping requested output names to TensorMaps of
            predictions, either per-atom or summed over atoms.
        """
        atomic_predictions_tmap_dict: Dict[str, TensorMap] = {}
        for output_name in self.target_names:
            if output_name in outputs:
                atomic_predictions_by_block = {
                    key: torch.zeros(
                        1, dtype=edge_vectors.dtype, device=edge_vectors.device
                    )
                    for key in self.output_shapes[output_name].keys()
                }

                node_atomic_predictions_by_block = node_atomic_predictions_dict[
                    output_name
                ]
                edge_atomic_predictions_by_block = edge_atomic_predictions_dict[
                    output_name
                ]
                for i in range(len(node_atomic_predictions_by_block)):
                    node_atomic_prediction_block = node_atomic_predictions_by_block[i]
                    edge_atomic_prediction_block = edge_atomic_predictions_by_block[i]
                    for j, key in enumerate(atomic_predictions_by_block):
                        node_atomic_predictions = node_atomic_prediction_block[j]
                        edge_atomic_predictions = edge_atomic_prediction_block[j]
                        atomic_predictions_by_block[key] = atomic_predictions_by_block[
                            key
                        ] + (node_atomic_predictions + edge_atomic_predictions)

                all_components = self.component_labels[output_name]
                if len(all_components[0]) == 2 and all(
                    "xyz" in comp.names[0] for comp in all_components[0]
                ):
                    block_key = list(atomic_predictions_by_block.keys())[0]
                    output_shapes_values = list(
                        self.output_shapes[output_name].values()
                    )
                    num_properties = output_shapes_values[0][-1]
                    symmetrized = symmetrize_cartesian_tensor(
                        atomic_predictions_by_block[block_key],
                        systems,
                        system_indices,
                        num_properties,
                    )
                    atomic_predictions_by_block[block_key] = symmetrized

                blocks = [
                    TensorBlock(
                        values=atomic_predictions_by_block[key].reshape([-1] + shape),
                        samples=sample_labels,
                        components=components,
                        properties=properties,
                    )
                    for key, shape, components, properties in zip(
                        self.output_shapes[output_name].keys(),
                        self.output_shapes[output_name].values(),
                        self.component_labels[output_name],
                        self.property_labels[output_name],
                        strict=True,
                    )
                ]
                atomic_predictions_tmap_dict[output_name] = TensorMap(
                    keys=self.key_labels[output_name],
                    blocks=blocks,
                )
        # If selected atoms request is provided, we slice the atomic predictions
        # tensor maps to get the predictions for the selected atoms only.

        if selected_atoms is not None:
            for output_name, tmap in atomic_predictions_tmap_dict.items():
                atomic_predictions_tmap_dict[output_name] = mts.slice(
                    tmap, axis="samples", selection=selected_atoms
                )

        # If per-atom predictions are requested, we return the atomic predictions
        # tensor maps. Otherwise, we sum the atomic predictions over the atoms
        # to get the final per-structure predictions for each requested output.

        for output_name, atomic_property in atomic_predictions_tmap_dict.items():
            if outputs[output_name].per_atom:
                atomic_predictions_tmap_dict[output_name] = atomic_property
            else:
                atomic_predictions_tmap_dict[output_name] = sum_over_atoms(
                    atomic_property
                )

        return atomic_predictions_tmap_dict

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> "PET":
        if context == "restart":
            logging.info(f"Using latest model from epoch {checkpoint['epoch']}")
            model_state_dict = checkpoint["model_state_dict"]
        elif context in {"finetune", "export"}:
            logging.info(f"Using best model from epoch {checkpoint['best_epoch']}")
            model_state_dict = checkpoint["best_model_state_dict"]
        else:
            raise ValueError("Unknown context tag for checkpoint loading!")

        # Create the model
        model_data = checkpoint["model_data"]
        model = cls(
            hypers=model_data["model_hypers"],
            dataset_info=model_data["dataset_info"],
        )

        finetune_config = model_state_dict.pop("finetune_config", {})
        if finetune_config:
            # Apply the finetuning strategy
            model = apply_finetuning_strategy(model, finetune_config)
        state_dict_iter = iter(model_state_dict.values())
        next(state_dict_iter)  # skip the species_to_species_index
        dtype = next(state_dict_iter).dtype
        model.to(dtype).load_state_dict(model_state_dict)
        model.additive_models[0].sync_tensor_maps()

        # Loading the metadata from the checkpoint
        model.metadata = merge_metadata(model.metadata, checkpoint.get("metadata"))

        return model

    def export(self, metadata: Optional[ModelMetadata] = None) -> AtomisticModel:
        dtype = next(self.parameters()).dtype
        if dtype not in self.__supported_dtypes__:
            raise ValueError(f"unsupported dtype {dtype} for PET")

        # Make sure the model is all in the same dtype
        # For example, after training, the additive models could still be in
        # float64
        self.to(dtype)

        # Additionally, the composition model contains some `TensorMap`s that cannot
        # be registered correctly with Pytorch. This function moves them:
        self.additive_models[0].weights_to(torch.device("cpu"), torch.float64)

        interaction_ranges = [self.num_gnn_layers * self.cutoff]
        for additive_model in self.additive_models:
            if hasattr(additive_model, "cutoff_radius"):
                interaction_ranges.append(additive_model.cutoff_radius)
        interaction_range = max(interaction_ranges)

        capabilities = ModelCapabilities(
            outputs=self.outputs,
            atomic_types=self.atomic_types,
            interaction_range=interaction_range,
            length_unit=self.dataset_info.length_unit,
            supported_devices=self.__supported_devices__,
            dtype=dtype_to_str(dtype),
        )

        metadata = merge_metadata(self.metadata, metadata)

        return AtomisticModel(self.eval(), metadata, capabilities)

    def _add_output(self, target_name: str, target_info: TargetInfo) -> None:
        """
        Register a new output target by creating corresponding heads and last layers.
        Sets up node/edge heads and linear layers for all readout layers.

        :param target_name: Name of the target to add.
        :param target_info: TargetInfo object containing details about the target.
        """
        # warn that, for Cartesian tensors, we assume that they are symmetric
        if target_info.is_cartesian:
            if len(target_info.layout.block().components) == 2:
                warnings.warn(
                    "PET assumes that Cartesian tensors of rank 2 are "
                    "stress-like, meaning that they are symmetric and intensive. "
                    "If this is not the case, please use a different model.",
                    UserWarning,
                    stacklevel=2,
                )
            # error out for rank > 2
            if len(target_info.layout.block().components) > 2:
                raise ValueError(
                    "PET does not support Cartesian tensors with rank > 2."
                )

        # one output shape for each tensor block, grouped by target (i.e. tensormap)
        self.output_shapes[target_name] = {}
        for key, block in target_info.layout.items():
            dict_key = target_name
            for n, k in zip(key.names, key.values, strict=True):
                dict_key += f"_{n}_{int(k)}"
            self.output_shapes[target_name][dict_key] = [
                len(comp.values) for comp in block.components
            ] + [len(block.properties.values)]

        self.outputs[target_name] = ModelOutput(
            quantity=target_info.quantity,
            unit=target_info.unit,
            per_atom=True,
        )

        self.node_heads[target_name] = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(self.d_node, self.d_head),
                    torch.nn.SiLU(),
                    torch.nn.Linear(self.d_head, self.d_head),
                    torch.nn.SiLU(),
                )
                for _ in range(self.num_readout_layers)
            ]
        )

        self.edge_heads[target_name] = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(self.d_pet, self.d_head),
                    torch.nn.SiLU(),
                    torch.nn.Linear(self.d_head, self.d_head),
                    torch.nn.SiLU(),
                )
                for _ in range(self.num_readout_layers)
            ]
        )

        self.node_last_layers[target_name] = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        key: torch.nn.Linear(
                            self.d_head,
                            prod(shape),
                            bias=True,
                        )
                        for key, shape in self.output_shapes[target_name].items()
                    }
                )
                for _ in range(self.num_readout_layers)
            ]
        )

        self.edge_last_layers[target_name] = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        key: torch.nn.Linear(
                            self.d_head,
                            prod(shape),
                            bias=True,
                        )
                        for key, shape in self.output_shapes[target_name].items()
                    }
                )
                for _ in range(self.num_readout_layers)
            ]
        )

        ll_features_name = get_last_layer_features_name(target_name)
        self.outputs[ll_features_name] = ModelOutput(per_atom=True)
        self.key_labels[target_name] = target_info.layout.keys
        self.component_labels[target_name] = [
            block.components for block in target_info.layout.blocks()
        ]
        self.property_labels[target_name] = [
            block.properties for block in target_info.layout.blocks()
        ]

    def _move_labels_to_device(self, device: torch.device) -> None:
        self.single_label = self.single_label.to(device)
        self.key_labels = {
            output_name: label.to(device)
            for output_name, label in self.key_labels.items()
        }
        self.component_labels = {
            output_name: [
                [labels.to(device) for labels in components_block]
                for components_block in components_tmap
            ]
            for output_name, components_tmap in self.component_labels.items()
        }
        self.property_labels = {
            output_name: [labels.to(device) for labels in properties_tmap]
            for output_name, properties_tmap in self.property_labels.items()
        }

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict) -> Dict:
        for v in range(1, cls.__checkpoint_version__):
            if checkpoint["model_ckpt_version"] == v:
                update = getattr(checkpoints, f"model_update_v{v}_v{v + 1}")
                update(checkpoint)
                checkpoint["model_ckpt_version"] = v + 1

        if checkpoint["model_ckpt_version"] != cls.__checkpoint_version__:
            raise RuntimeError(
                f"Unable to upgrade the checkpoint: the checkpoint is using model "
                f"version {checkpoint['model_ckpt_version']}, while the current model "
                f"version is {cls.__checkpoint_version__}."
            )

        return checkpoint

    def get_checkpoint(self) -> Dict:
        model_state_dict = self.state_dict()
        model_state_dict["finetune_config"] = self.finetune_config
        checkpoint = {
            "architecture_name": "pet",
            "model_ckpt_version": self.__checkpoint_version__,
            "metadata": self.metadata,
            "model_data": {
                "model_hypers": self.hypers,
                "dataset_info": self.dataset_info,
            },
            "epoch": None,
            "best_epoch": None,
            "model_state_dict": model_state_dict,
            "best_model_state_dict": self.state_dict(),
        }
        return checkpoint


def symmetrize_cartesian_tensor(
    tensor: torch.Tensor,
    systems: List[System],
    system_indices: torch.Tensor,
    num_properties: int,
) -> torch.Tensor:
    """
    Symmetrize rank-2 Cartesian tensors (e.g., stress).
    Assumes the tensor is stress-like (symmetric and intensive).

    :param tensor: Tensor of shape [n_atoms, 9 * num_properties].
    :param systems: List of `metatomic.torch.System` objects to process.
    :param system_indices: Tensor mapping each atom to its system index [n_atoms].
    :param num_properties: Number of properties in the tensor (e.g., 6 for stress).
    :return: Symmetrized tensor of shape [n_atoms, 3, 3, num_properties].
    """
    # Reshape to 3x3 matrix per atom
    tensor_as_three_by_three = tensor.reshape(-1, 3, 3, num_properties)

    # Normalize by cell volume
    volumes = torch.stack([torch.abs(torch.det(system.cell)) for system in systems])
    volumes_by_atom = volumes[system_indices].unsqueeze(1).unsqueeze(2).unsqueeze(3)
    tensor_as_three_by_three = tensor_as_three_by_three / volumes_by_atom

    # Symmetrize
    tensor_as_three_by_three = (
        tensor_as_three_by_three + tensor_as_three_by_three.transpose(1, 2)
    ) / 2.0

    return tensor_as_three_by_three


def get_last_layer_features_name(target_name: str) -> str:
    """
    Get the auxiliary output name for last layer features of a target.

    :param target_name: Name of the target.
    :return: Name of the corresponding last layer features output.
    """
    base_name = target_name.replace("mtt::", "")
    return f"mtt::aux::{base_name}_last_layer_features"


def should_compute_last_layer_features(
    output_name: str, requested_outputs: Dict[str, ModelOutput]
) -> bool:
    """
    Check if last layer features should be computed for an output.

    :param output_name: Name of the output to check.
    :param requested_outputs: Dictionary of requested outputs.
    :return: True if last layer features should be computed, False otherwise.
    """
    if output_name in requested_outputs:
        return True
    ll_features_name = get_last_layer_features_name(
        output_name.replace("mtt::aux::", "")
    )
    return ll_features_name in requested_outputs
