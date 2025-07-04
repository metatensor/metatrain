import warnings
from math import prod
from typing import Any, Dict, List, Literal, Optional

import metatensor.torch
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
from metatrain.utils.basis import (
    extract_key_value,
    get_edge_sample_labels,
    get_permutation_symmetrization_arrays,
    get_sample_labels_block,
    get_system_indices_and_node_sample_labels,
)
from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.dtype import dtype_to_str
from metatrain.utils.long_range import DummyLongRangeFeaturizer, LongRangeFeaturizer
from metatrain.utils.metadata import merge_metadata
from metatrain.utils.scaler import Scaler
from metatrain.utils.sum_over_atoms import sum_over_atoms

from .modules.finetuning import apply_finetuning_strategy
from .modules.structures import remap_neighborlists, systems_to_batch
from .modules.transformer import CartesianTransformer
from .modules.utilities import cutoff_func


class PET(ModelInterface):
    """
    Metatrain-native implementation of the PET architecture.

    Originally proposed in work (https://arxiv.org/abs/2305.19302v3),
    and published in the `pet` package (https://github.com/spozdn/pet).

    """

    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float32, torch.float64]
    __default_metadata__ = ModelMetadata(
        references={"architecture": ["https://arxiv.org/abs/2305.19302v3"]}
    )
    component_labels: Dict[str, List[List[Labels]]]

    def __init__(self, model_hypers: Dict, dataset_info: DatasetInfo) -> None:
        super().__init__()
        self.dataset_info = dataset_info
        self.atomic_types = dataset_info.atomic_types
        self.hypers = model_hypers
        self.requested_nl = NeighborListOptions(
            cutoff=self.hypers["cutoff"],
            full_list=True,
            strict=True,
        )

        self.cutoff = float(self.hypers["cutoff"])
        self.cutoff_width = float(self.hypers["cutoff_width"])
        self.embedding = torch.nn.Embedding(
            len(self.atomic_types) + 1, self.hypers["d_pet"]
        )
        gnn_layers = []
        for layer_index in range(self.hypers["num_gnn_layers"]):
            transformer_layer = CartesianTransformer(
                self.hypers,
                self.hypers["d_pet"],
                self.hypers["num_heads"],
                self.hypers["d_feedforward"],
                self.hypers["num_attention_layers"],
                0.0,  # attention dropout rate
                len(self.atomic_types),
                layer_index == 0,  # is first layer
            )
            gnn_layers.append(transformer_layer)

        self.gnn_layers = torch.nn.ModuleList(gnn_layers)

        self.node_heads = torch.nn.ModuleDict()
        self.edge_heads = torch.nn.ModuleDict()
        self.node_last_layers = torch.nn.ModuleDict()
        self.edge_last_layers = torch.nn.ModuleDict()
        self.last_layer_feature_size = (
            self.hypers["num_gnn_layers"] * self.hypers["d_head"] * 2
        )

        self.outputs = {
            "features": ModelOutput(unit="", per_atom=True)
        }  # the model is always capable of outputting the internal features

        self.output_shapes: Dict[str, Dict[str, List[int]]] = {}
        self.key_labels: Dict[str, Labels] = {}
        self.property_labels: Dict[str, List[Labels]] = {}
        self.component_labels: Dict[str, List[List[Labels]]] = {}
        self.target_names: List[str] = []
        self.sample_kinds: Dict[str, str] = {}
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
                feature_dim=self.hypers["d_pet"],
                neighbor_list_options=self.requested_nl,
            )
        else:
            self.long_range = False
            self.long_range_featurizer = DummyLongRangeFeaturizer()  # for torchscript

        # additive models: these are handled by the trainer at training
        # time, and they are added to the output at evaluation time
        composition_model = CompositionModel(
            model_hypers={},
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
        self.scaler = Scaler(model_hypers={}, dataset_info=dataset_info)

        self.single_label = Labels.single()

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

    def requested_neighbor_lists(
        self,
    ) -> List[NeighborListOptions]:
        return [self.requested_nl]

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        device = systems[0].device
        return_dict: Dict[str, TensorMap] = {}
        nl_options = self.requested_neighbor_lists()[0]

        if not self.training:
            # While running the model with LAMMPS, we need to remap the
            # neighbor lists from LAMMPS to ASE format. By default, LAMMPS
            # treats all ghost atoms as real (central), what creates a
            # singificant computational overhead.
            systems = remap_neighborlists(systems, nl_options, selected_atoms)

        if self.single_label.values.device != device:
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

        system_indices, node_sample_labels = get_system_indices_and_node_sample_labels(
            systems, device
        )
        if any([kind == "per_pair" for kind in self.sample_kinds.values()]):
            (
                edge_sample_labels_1_center,
                edge_sample_labels_2_center,
            ) = get_edge_sample_labels(systems, node_sample_labels, nl_options, device)
        else:
            (
                edge_sample_labels_1_center,
                edge_sample_labels_2_center,
            ) = (
                Labels("_", torch.empty(0).reshape(-1, 1)),
                Labels("_", torch.empty(0).reshape(-1, 1)),
            )

        # We convert a list of systems to a batch required for the PET model.
        # The batch consists of the following tensors: f
        # - `element_indices_nodes` [n_atoms]: The atomic species of the central atoms
        # - `element_indices_neighbors` [n_atoms]: The atomic species of the neighboring
        #   atoms
        # - `edge_vectors` [n_atoms, max_num_neighbors, 3]: The cartedian edge vectors
        #   between the central atoms and their neighbors
        # - `padding_mask` [n_atoms, max_num_neighbors]: A padding mask indicating which
        #   neighbors are real, and which are padded
        # - `neighbors_index` [n_atoms, max_num_neighbors]: The indices of the
        #   neighboring atoms for each central atom
        # - `num_neghbors` [n_atoms]: The number of neighbors for each central atom
        # - `reversed_neighbor_list` [n_atoms, max_num_neighbors]: The reversed neighbor
        #   list for each central atom, where for each center atom `i` and its neighbor
        #   `j` in the original neighborlist, the position of atom `i` in the list of
        #   neighbors of atom `j` is returned.

        (
            element_indices_nodes,
            element_indices_neighbors,
            edge_vectors,
            padding_mask,
            neighbors_index,
            num_neghbors,
            reversed_neighbor_list,
        ) = systems_to_batch(
            systems,
            nl_options,
            self.atomic_types,
            system_indices,
            self.species_to_species_index,
            selected_atoms,
        )

        # if we are predicting per-pair outputs that are permutationally symmetrized, we
        # need some samples masks for edges with different and same atom types, and a
        # map to permute the samples of the same atom types

        if any(["s2_pi" in keys.names for keys in self.key_labels.values()]):
            (
                samples_mask_2_center_same_types,
                samples_mask_2_center_diff_types,
                permuted_samples_map_same_types,
                edge_sample_labels_2_center_same_types,
                edge_sample_labels_2_center_diff_types,
            ) = get_permutation_symmetrization_arrays(systems, edge_sample_labels_2_center)
        else:
            (
                edge_sample_labels_2_center_same_types,
                edge_sample_labels_2_center_diff_types,
            ) = (
                Labels("_", torch.empty(0).reshape(-1, 1)),
                Labels("_", torch.empty(0).reshape(-1, 1)),
            )

        # the scaled_dot_product_attention function from torch cannot do
        # double backward, so we will use manual attention if needed
        use_manual_attention = edge_vectors.requires_grad and self.training

        edge_distances = torch.sqrt(torch.sum(edge_vectors**2, dim=2) + 1e-15)
        cutoff_factors = cutoff_func(edge_distances, self.cutoff, self.cutoff_width)
        cutoff_factors[~padding_mask] = 0.0

        # Stage 1. We iterate over the GNN layers and calculate the node and edge
        # representations for structures, while saving the intermediate node and edge
        # features from each layer to the corresponding lists.

        node_features_list: List[torch.Tensor] = []
        edge_features_list: List[torch.Tensor] = []

        input_messages = self.embedding(element_indices_neighbors)
        for gnn_layer in self.gnn_layers:
            output_node_embeddings, output_edge_embeddings = gnn_layer(
                input_messages,
                element_indices_nodes,
                element_indices_neighbors,
                edge_vectors,
                padding_mask,
                edge_distances,
                cutoff_factors,
                use_manual_attention,
            )
            node_features_list.append(output_node_embeddings)
            edge_features_list.append(output_edge_embeddings)

            # The GNN contraction happens by reordering the messages,
            # using a reversed neighbor list, so the new input message
            # from atom `j` to atom `i` in on the GNN layer N+1 is a
            # reversed message from atom `i` to atom `j` on the GNN layer N.
            new_input_messages = output_edge_embeddings[
                neighbors_index, reversed_neighbor_list
            ]
            input_messages = 0.5 * (input_messages + new_input_messages)

        # If the long-range module is actuvated, we add the long-range features
        # on top of the node features

        if self.long_range:
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
            for i in range(len(self.gnn_layers)):
                node_features_list[i] = (
                    node_features_list[i] + long_range_features
                ) * 0.5**0.5

        # Stage 2. If `features` requested in the model outputs, we concatenate
        # the node and edge representations from all layers to provide the intermediate
        # representation of the systems. Since edge features are calculated for each
        # pair of atoms, we sum them up with cutoff factors to get their per-node
        # contribution.

        if "features" in outputs:
            node_features = torch.cat(node_features_list, dim=1)
            edge_features = torch.cat(edge_features_list, dim=2)
            edge_features = edge_features * cutoff_factors[:, :, None]
            edge_features = edge_features.sum(dim=1)
            features = torch.cat([node_features, edge_features], dim=1)

            feature_tmap = TensorMap(
                keys=self.single_label,
                blocks=[
                    TensorBlock(
                        values=features,
                        samples=node_sample_labels,
                        components=[],
                        properties=Labels(
                            names=["properties"],
                            values=torch.arange(
                                features.shape[-1], device=features.device
                            ).reshape(-1, 1),
                        ),
                    )
                ],
            )
            features_options = outputs["features"]
            if features_options.per_atom:
                return_dict["features"] = feature_tmap
            else:
                return_dict["features"] = sum_over_atoms(feature_tmap)

        # Stage 3. We compute last layer features for each requested output,
        # for both node and edge features from each GNN layer. To do this, apply the
        # corresponding heads to both node and edge features, and save the results
        # to the corresponsing dicts. Finally, we stack all the last layer features
        # to get the final last-layer-features tensor.

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

        # Stacking node and edge last layer features to get the final
        # last-layer-features tensor. As was done earlier to `features`
        # tensor, we sum the edge features with cutoff factors to get their
        # per-node contribution.

        last_layer_features_dict: Dict[str, List[torch.Tensor]] = {}
        for output_name in self.target_names:
            if output_name not in last_layer_features_dict:
                last_layer_features_dict[output_name] = []
            for i in range(len(node_last_layer_features_dict[output_name])):
                node_last_layer_features = node_last_layer_features_dict[output_name][i]
                edge_last_layer_features = edge_last_layer_features_dict[output_name][i]
                edge_last_layer_features = (
                    edge_last_layer_features * cutoff_factors[:, :, None]
                )
                edge_last_layer_features = edge_last_layer_features.sum(dim=1)
                last_layer_features_dict[output_name].append(node_last_layer_features)
                last_layer_features_dict[output_name].append(edge_last_layer_features)

        for output_name in outputs.keys():
            if not (
                output_name.startswith("mtt::aux::")
                and output_name.endswith("_last_layer_features")
            ):
                continue
            base_name = output_name.replace("mtt::aux::", "").replace(
                "_last_layer_features", ""
            )
            # the corresponding output could be base_name or mtt::base_name
            if (
                f"mtt::{base_name}" not in last_layer_features_dict
                and base_name not in last_layer_features_dict
            ):
                raise ValueError(
                    f"Features {output_name} can only be requested "
                    f"if the corresponding output {base_name} is also requested."
                )
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
                        samples=node_sample_labels,
                        components=[],
                        properties=Labels(
                            names=["properties"],
                            values=torch.arange(
                                last_layer_features_values.shape[-1],
                                device=last_layer_features_values.device,
                            ).reshape(-1, 1),
                        ),
                    )
                ],
            )
            last_layer_features_options = outputs[output_name]
            if last_layer_features_options.per_atom:
                return_dict[output_name] = last_layer_feature_tmap
            else:
                return_dict[output_name] = sum_over_atoms(last_layer_feature_tmap)

        # Stage 4. We compute the per-atom predictions by applying the
        # linear layers to both node and edge last layer features. To do this,
        # we iterate over the last layer features (both node and edge), and
        # apply the corresponding last layer to each feature for each requested
        # output.

        atomic_predictions_tmap_dict: Dict[str, TensorMap] = {}

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
                    for key, node_last_layer_by_block in node_last_layer.items():

                        # depending on the block key, the edge predictions are handled
                        # in different ways:
                        #
                        # 1) if the block key does not contain an "n_centers" dimension,
                        #    the last layer node features are transformed into
                        #    predictions by means of the node last (linear) layer
                        #
                        # 2) if the block key contains an "n_centers" dimension and it
                        #    is equal to 1, the node features are treated the same as in
                        #    1).
                        #
                        # 3) if the "n_centers" dimension is present and is equal to 2
                        #    this block is a per-pair target. Node predictions are not
                        #    computed for these blocks, so a zero-sample tensor with the
                        #    correct feature size is stored.

                        if "n_centers" not in key:  # 1)
                            node_atomic_predictions_by_block.append(
                                node_last_layer_by_block(node_last_layer_features)
                            )

                        else:
                            if extract_key_value(key, "n_centers") == 1:  # 2)
                                node_atomic_predictions_by_block.append(
                                    node_last_layer_by_block(node_last_layer_features)
                                )

                            else:  # 3)
                                assert extract_key_value(key, "n_centers") == 2
                                node_atomic_predictions_by_block.append(
                                    torch.empty(
                                        (0, prod(self.output_shapes[output_name][key])),
                                        dtype=node_last_layer_features.dtype,
                                        device=node_last_layer_features.device,
                                    ),
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
                    edge_atomic_predictions_by_block: List[torch.Tensor] = []

                    for key, edge_last_layer_by_block in edge_last_layer.items():

                        # for each GNN layer, the edge last layer features are
                        # transformed into block predictions by means of the
                        # corresponding edge last (linear) layer. Then, predictions for
                        # edges not in the neighbor list are masked out and the cutoff
                        # function is applied.

                        edge_last_layer_features = edge_last_layer_features_dict[output_name][i]
                        edge_atomic_predictions = edge_last_layer_by_block(
                            edge_last_layer_features
                        )
                        expanded_padding_mask = padding_mask[..., None].repeat(
                            1, 1, edge_atomic_predictions.shape[2]
                        )
                        edge_atomic_predictions = torch.where(
                            ~expanded_padding_mask, 0.0, edge_atomic_predictions
                        )
                        edge_atomic_predictions = (
                            edge_atomic_predictions * cutoff_factors[:, :, None]
                        )

                        # depending on the block key, these edge predictions are then
                        # handled in different ways.
                        #
                        # 1) if the block key does not contain an "n_centers" dimension,
                        #    the edge predictions are summed over the neighbors and
                        #    added (later) to the node predictions.
                        #
                        # 2) if the block key contains an "n_centers" dimension and it
                        #    is equal to 1, the edge predictions are treated as in 1).
                        #
                        # 3) if the "n_centers" dimension is present and is equal to 2
                        #    this block is a per-pair target. The edge predictions are
                        #    reshaped to contain both the central and neighbor atoms
                        #    along the samples axis. The padding mask is used to slice
                        #    out the samples not in the actual neighbor list. Then, the
                        #    permutation-symmetrization protocol is applied as follows,
                        #    depending on the value of the "s2_pi" key dimension. If
                        #    "s2_pi" is 0, the samples are sliced to atom pairs of
                        #    different atomic types. If "s2_pi" is +1 or -1, the samples
                        #    are sliced to atom pairs of the same atomic type, and
                        #    symmetrization is performed by taking the plus or minus
                        #    combination (respectively) of the samples.

                        if "n_centers" not in key:  # 1)
                            edge_atomic_predictions = edge_atomic_predictions.sum(dim=1)

                        else:
                            if extract_key_value(key, "n_centers") == 1:  # 2)
                                edge_atomic_predictions = edge_atomic_predictions.sum(dim=1)

                            else:  # 3)

                                assert extract_key_value(key, "n_centers") == 2

                                # reshape the edge predictions to have both central
                                # atoms and neighbors in the samples axis and slice out
                                # the samples not in the actual neighbor list

                                edge_atomic_predictions = edge_atomic_predictions.reshape(
                                    -1, edge_atomic_predictions.shape[-1]
                                )
                                edge_atomic_predictions = edge_atomic_predictions[
                                    padding_mask.reshape(-1)
                                ]

                                if "s2_pi" in key:

                                    # if s2_pi == 0, slice the edge predictions to atom
                                    # pairs of different atomic types. No symmetrization is
                                    # required.

                                    if extract_key_value(key, "s2_pi") == 0:
                                        edge_atomic_predictions = edge_atomic_predictions[
                                            samples_mask_2_center_diff_types
                                        ]

                                    # otherwise if s2_pi = +/- 1, slice to atom pairs of
                                    # different types and permutation-symmetrize depending
                                    # on the value of s2_pi.

                                    else:  # same atom type
                                        s2_pi = extract_key_value(key, "s2_pi")
                                        assert s2_pi in [1, -1]
                                        edge_atomic_predictions = edge_atomic_predictions[
                                            samples_mask_2_center_same_types
                                        ]
                                        edge_atomic_predictions = (
                                            edge_atomic_predictions 
                                            + (
                                                s2_pi 
                                                * edge_atomic_predictions[
                                                    permuted_samples_map_same_types
                                                ]
                                            )
                                        ) / 2.0  # TODO: do we want this factor of 2?

                        edge_atomic_predictions_by_block.append(edge_atomic_predictions)
                    edge_atomic_predictions_dict[output_name].append(
                        edge_atomic_predictions_by_block
                    )

        # Finally, we sum all the node and edge atomic predictions from each GNN
        # layer to a single atomic predictions tensor.

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

                        # depending on the key, the node and edge predictions are
                        # handled differently:
                        #
                        # 1) if the block key does not contain an "n_centers" dimension,
                        #    the node and edge predictions are the same shape and are
                        #    summed together and accumulated across GNN layers.
                        #
                        # 2) if the block key contains an "n_centers" dimension and it
                        #    is equal to 1, the node and edge predictions are treated
                        #    the same as in 1).
                        #
                        # 3) if the "n_centers" dimension is present and is equal to 2
                        #    this block is a per-pair target. The node predictions are
                        #    empty are not used, and only the edge predictions are
                        #    accumulated across GNN layers.

                        if "n_centers" not in key:  # 1)
                            atomic_predictions_by_block[key] = atomic_predictions_by_block[
                                key
                            ] + (node_atomic_predictions + edge_atomic_predictions)

                        else:
                            if extract_key_value(key, "n_centers") == 1:  # 2)
                                atomic_predictions_by_block[key] = (
                                    atomic_predictions_by_block[key]
                                    + (node_atomic_predictions + edge_atomic_predictions)
                                )

                            else:  # 3)
                                assert extract_key_value(key, "n_centers") == 2
                                assert node_atomic_predictions.shape[0] == 0
                                atomic_predictions_by_block[key] = (
                                    atomic_predictions_by_block[key]
                                    + edge_atomic_predictions
                                )

                all_components = self.component_labels[output_name]
                if len(all_components[0]) == 2 and all(
                    "xyz" in comp.names[0] for comp in all_components[0]
                ):
                    block_key = list(atomic_predictions_by_block.keys())[0]
                    # rank-2 Cartesian tensor, symmetrize
                    tensor_as_three_by_three = atomic_predictions_by_block[
                        block_key
                    ].reshape(
                        -1, 3, 3, list(self.output_shapes[output_name].values())[0][-1]
                    )
                    volumes = torch.stack(
                        [torch.abs(torch.det(system.cell)) for system in systems]
                    )
                    volumes_by_atom = (
                        volumes[system_indices].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    )
                    tensor_as_three_by_three = (
                        tensor_as_three_by_three / volumes_by_atom
                    )
                    tensor_as_three_by_three = (
                        tensor_as_three_by_three
                        + tensor_as_three_by_three.transpose(1, 2)
                    ) / 2.0
                    atomic_predictions_by_block[block_key] = tensor_as_three_by_three

                blocks = [
                    TensorBlock(
                        values=atomic_predictions_by_block[key].reshape([-1] + shape),
                        samples=get_sample_labels_block(
                            key,
                            self.sample_kinds[output_name],
                            node_sample_labels,
                            edge_sample_labels_1_center,
                            edge_sample_labels_2_center,
                            edge_sample_labels_2_center_same_types,
                            edge_sample_labels_2_center_diff_types,
                        ),
                        components=components,
                        properties=properties,
                    )
                    for key, shape, components, properties in zip(
                        self.output_shapes[output_name].keys(),
                        self.output_shapes[output_name].values(),
                        self.component_labels[output_name],
                        self.property_labels[output_name],
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
                atomic_predictions_tmap_dict[output_name] = metatensor.torch.slice(
                    tmap, axis="samples", selection=selected_atoms
                )

        # If per-atom predictions are requested, we return the atomic predictions
        # tensor maps. Otherwise, we sum the atomic predictions over the atoms
        # to get the final per-structure predictions for each requested output.

        for output_name, atomic_property in atomic_predictions_tmap_dict.items():
            if outputs[output_name].per_atom:
                return_dict[output_name] = atomic_property
            else:
                return_dict[output_name] = sum_over_atoms(atomic_property)

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

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> "PET":
        model_data = checkpoint["model_data"]

        if context == "restart":
            model_state_dict = checkpoint["model_state_dict"]
        elif context == "finetune" or context == "export":
            model_state_dict = checkpoint["best_model_state_dict"]
        else:
            raise ValueError("Unknown context tag for checkpoint loading!")

        finetune_config = checkpoint["train_hypers"].get("finetune", {})

        # Create the model
        model = cls(**model_data)

        if finetune_config:
            # Apply the finetuning strategy
            model = apply_finetuning_strategy(model, finetune_config)
        state_dict_iter = iter(model_state_dict.values())
        next(state_dict_iter)  # skip the species_to_species_index
        dtype = next(state_dict_iter).dtype
        model.to(dtype).load_state_dict(model_state_dict)
        model.additive_models[0].sync_tensor_maps()

        # Loading the metadata from the checkpoint
        metadata = checkpoint.get("metadata", None)
        if metadata is not None:
            model.__default_metadata__ = metadata

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
        self.additive_models[0]._move_weights_to_device_and_dtype(
            torch.device("cpu"), torch.float64
        )

        interaction_ranges = [self.hypers["num_gnn_layers"] * self.hypers["cutoff"]]
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

        if metadata is None:
            metadata = self.__default_metadata__
        else:
            metadata = merge_metadata(self.__default_metadata__, metadata)

        return AtomisticModel(self.eval(), metadata, capabilities)

    def _add_output(self, target_name: str, target_info: TargetInfo) -> None:
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
            for n, k in zip(key.names, key.values):
                dict_key += f"_{n}_{int(k)}"
            self.output_shapes[target_name][dict_key] = [
                len(comp.values) for comp in block.components
            ] + [len(block.properties.values)]

        self.outputs[target_name] = ModelOutput(
            quantity=target_info.quantity,
            unit=target_info.unit,
            per_atom=True,
        )
        # store the sample kind
        if target_info.layout.sample_names == ["system"]:
            self.sample_kinds[target_name] = "per_structure"
        elif target_info.layout.sample_names == ["system", "atom"]:
            self.sample_kinds[target_name] = "per_atom"
        else:
            assert target_info.layout.sample_names == [
                "system",
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ]
            self.sample_kinds[target_name] = "per_pair"

        self.node_heads[target_name] = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(self.hypers["d_pet"], self.hypers["d_head"]),
                    torch.nn.SiLU(),
                    torch.nn.Linear(self.hypers["d_head"], self.hypers["d_head"]),
                    torch.nn.SiLU(),
                )
                for _ in range(self.hypers["num_gnn_layers"])
            ]
        )

        self.edge_heads[target_name] = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(self.hypers["d_pet"], self.hypers["d_head"]),
                    torch.nn.SiLU(),
                    torch.nn.Linear(self.hypers["d_head"], self.hypers["d_head"]),
                    torch.nn.SiLU(),
                )
                for _ in range(self.hypers["num_gnn_layers"])
            ]
        )

        self.node_last_layers[target_name] = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        key: self._init_node_last_layer_by_key(key, shape)
                        for key, shape in self.output_shapes[target_name].items()
                    }
                )
                for _ in range(self.hypers["num_gnn_layers"])
            ]
        )

        self.edge_last_layers[target_name] = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        key: torch.nn.Linear(
                            self.hypers["d_head"],
                            prod(shape),
                            bias=True,
                        )
                        for key, shape in self.output_shapes[target_name].items()
                    }
                )
                for _ in range(self.hypers["num_gnn_layers"])
            ]
        )

        ll_features_name = (
            f"mtt::aux::{target_name.replace('mtt::', '')}_last_layer_features"
        )
        self.outputs[ll_features_name] = ModelOutput(per_atom=True)
        self.key_labels[target_name] = target_info.layout.keys
        self.component_labels[target_name] = [
            block.components for block in target_info.layout.blocks()
        ]
        self.property_labels[target_name] = [
            block.properties for block in target_info.layout.blocks()
        ]

    def _init_node_last_layer_by_key(
        self, key, shape,
    ) -> torch.nn.Module:
        if "n_centers" not in key:
            return torch.nn.Linear(
                self.hypers["d_head"],
                prod(shape),
                bias=True,
            )
        elif "n_centers" in key and extract_key_value(key, "n_centers") == 1:
            return torch.nn.Linear(
                self.hypers["d_head"],
                prod(shape),
                bias=True,
            )

        return torch.nn.Identity()
