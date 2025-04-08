from math import prod
from pathlib import Path
from typing import Dict, List, Optional, Union

import metatensor.torch
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)

from ...utils.additive import ZBL, CompositionModel
from ...utils.data import DatasetInfo, TargetInfo
from ...utils.dtype import dtype_to_str
from ...utils.long_range import DummyLongRangeFeaturizer, LongRangeFeaturizer
from ...utils.metadata import append_metadata_references
from ...utils.scaler import Scaler
from ...utils.sum_over_atoms import sum_over_atoms
from .modules.encoder import Encoder
from .modules.nef import (
    edge_array_to_nef,
    get_corresponding_edges,
    get_nef_indices,
    nef_array_to_edges,
)
from .modules.radial_mask import get_radial_mask
from .modules.structures import concatenate_structures
from .modules.transformer import Transformer


class NanoPET(torch.nn.Module):
    """
    Re-implementation of the PET architecture (https://arxiv.org/pdf/2305.19302).

    The positions and atomic species are encoded into a high-dimensional space
    using a simple encoder. The resulting features (in NEF, or Node-Edge-Feature
    format*) are then processed by a series of transformer layers. This process is
    repeated for a number of message-passing layers, where features are exchanged
    between corresponding edges (ij and ji). The final representation is used to
    predict atomic properties through decoders named "heads".

    * NEF format: a three-dimensional tensor where the first dimension corresponds
    to the nodes, the second to the edges corresponding to the neighbors of the
    node (padded as different nodes might have different numbers of edges),
    and the third to the features.
    """

    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float64, torch.float32]
    __default_metadata__ = ModelMetadata(
        references={"architecture": ["https://arxiv.org/abs/2305.19302v3"]}
    )

    component_labels: Dict[str, List[List[Labels]]]

    def __init__(self, model_hypers: Dict, dataset_info: DatasetInfo) -> None:
        super().__init__()
        # checks on targets inside the RotationalAugmenter class in the trainer

        self.hypers = model_hypers
        self.dataset_info = dataset_info
        self.new_outputs = list(dataset_info.targets.keys())
        self.atomic_types = dataset_info.atomic_types

        self.requested_nl = NeighborListOptions(
            cutoff=self.hypers["cutoff"],
            full_list=True,
            strict=True,
        )

        self.cutoff = float(self.hypers["cutoff"])
        self.cutoff_width = float(self.hypers["cutoff_width"])

        self.encoder = Encoder(len(self.atomic_types), self.hypers["d_pet"])

        self.transformer = Transformer(
            self.hypers["d_pet"],
            4 * self.hypers["d_pet"],
            self.hypers["num_heads"],
            self.hypers["num_attention_layers"],
            0.0,  # MLP dropout rate
            0.0,  # attention dropout rate
        )
        # empirically, the model seems to perform better without dropout

        self.num_mp_layers = self.hypers["num_gnn_layers"] - 1
        gnn_contractions = []
        gnn_transformers = []
        for _ in range(self.num_mp_layers):
            gnn_contractions.append(
                torch.nn.Linear(
                    2 * self.hypers["d_pet"], self.hypers["d_pet"], bias=False
                )
            )
            gnn_transformers.append(
                Transformer(
                    self.hypers["d_pet"],
                    4 * self.hypers["d_pet"],
                    self.hypers["num_heads"],
                    self.hypers["num_attention_layers"],
                    0.0,  # MLP dropout rate
                    0.0,  # attention dropout rate
                )
            )
        self.gnn_contractions = torch.nn.ModuleList(gnn_contractions)
        self.gnn_transformers = torch.nn.ModuleList(gnn_transformers)

        self.last_layer_feature_size = self.hypers["d_pet"]

        self.outputs = {
            "features": ModelOutput(unit="", per_atom=True)
        }  # the model is always capable of outputting the internal features

        self.heads = torch.nn.ModuleDict()
        self.head_types = self.hypers["heads"]
        self.last_layers = torch.nn.ModuleDict()
        self.output_shapes: Dict[str, Dict[str, List[int]]] = {}
        self.key_labels: Dict[str, Labels] = {}
        self.component_labels: Dict[str, List[List[Labels]]] = {}
        self.property_labels: Dict[str, List[Labels]] = {}
        for target_name, target_info in dataset_info.targets.items():
            self._add_output(target_name, target_info)

        self.register_buffer(
            "species_to_species_index",
            torch.full(
                (max(self.atomic_types) + 1,),
                -1,
            ),
        )
        for i, species in enumerate(self.atomic_types):
            self.species_to_species_index[species] = i

        # long-range module
        if self.hypers["long_range"]["enable"]:
            self.long_range = True
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

    def restart(self, dataset_info: DatasetInfo) -> "NanoPET":
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

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        # Checks on systems (species) and outputs are done in the
        # MetatensorAtomisticModel wrapper

        device = systems[0].device

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

        system_indices = torch.concatenate(
            [
                torch.full(
                    (len(system),),
                    i_system,
                    device=device,
                )
                for i_system, system in enumerate(systems)
            ],
        )

        sample_values = torch.stack(
            [
                system_indices,
                torch.concatenate(
                    [
                        torch.arange(
                            len(system),
                            device=device,
                        )
                        for system in systems
                    ],
                ),
            ],
            dim=1,
        )
        sample_labels = Labels(
            names=["system", "atom"],
            values=sample_values,
        )

        (
            positions,
            centers,
            neighbors,
            species,
            cells,
            cell_shifts,
        ) = concatenate_structures(systems, self.requested_nl)

        # somehow the backward of this operation is very slow at evaluation,
        # where there is only one cell, therefore we simplify the calculation
        # for that case
        if len(cells) == 1:
            cell_contributions = cell_shifts.to(cells.dtype) @ cells[0]
        else:
            cell_contributions = torch.einsum(
                "ab, abc -> ac",
                cell_shifts.to(cells.dtype),
                cells[system_indices[centers]],
            )

        edge_vectors = positions[neighbors] - positions[centers] + cell_contributions

        bincount = torch.bincount(centers)
        if bincount.numel() == 0:  # no edges
            max_edges_per_node = 0
        else:
            max_edges_per_node = int(torch.max(bincount))

        # Convert to NEF (Node-Edge-Feature) format:
        nef_indices, nef_to_edges_neighbor, nef_mask = get_nef_indices(
            centers, len(positions), max_edges_per_node
        )

        # Get radial mask
        r = torch.sqrt(torch.sum(edge_vectors**2, dim=-1))
        radial_mask = get_radial_mask(r, self.cutoff, self.cutoff - self.cutoff_width)

        # Element indices
        element_indices_nodes = self.species_to_species_index[species]
        element_indices_centers = element_indices_nodes[centers]
        element_indices_neighbors = element_indices_nodes[neighbors]

        # Send everything to NEF:
        edge_vectors = edge_array_to_nef(edge_vectors, nef_indices)
        radial_mask = edge_array_to_nef(
            radial_mask, nef_indices, nef_mask, fill_value=0.0
        )
        element_indices_centers = edge_array_to_nef(
            element_indices_centers, nef_indices
        )
        element_indices_neighbors = edge_array_to_nef(
            element_indices_neighbors, nef_indices
        )

        features = {
            "cartesian": edge_vectors,
            "center": element_indices_centers,
            "neighbor": element_indices_neighbors,
        }

        # Encode
        features = self.encoder(features)

        # Transformer
        features = self.transformer(features, radial_mask)

        # GNN
        if self.num_mp_layers > 0:
            corresponding_edges = get_corresponding_edges(
                torch.concatenate(
                    [centers.unsqueeze(-1), neighbors.unsqueeze(-1), cell_shifts],
                    dim=-1,
                )
            )
            for contraction, transformer in zip(
                self.gnn_contractions, self.gnn_transformers
            ):
                new_features = nef_array_to_edges(
                    features, centers, nef_to_edges_neighbor
                )
                corresponding_new_features = new_features[corresponding_edges]
                new_features = torch.concatenate(
                    [new_features, corresponding_new_features], dim=-1
                )
                new_features = contraction(new_features)
                new_features = edge_array_to_nef(new_features, nef_indices)
                new_features = transformer(new_features, radial_mask)
                features = (features + new_features) * 0.5**0.5

        edge_features = features * radial_mask[:, :, None]
        node_features = torch.sum(edge_features, dim=1)

        if self.long_range:
            long_range_node_features = self.long_range_featurizer(
                systems, node_features, r
            )
            node_features = (node_features + long_range_node_features) * 0.5**0.5

        return_dict: Dict[str, TensorMap] = {}

        # output the hidden features, if requested:
        if "features" in outputs:
            feature_tmap = TensorMap(
                keys=self.single_label,
                blocks=[
                    TensorBlock(
                        values=node_features,
                        samples=sample_labels,
                        components=[],
                        properties=Labels(
                            names=["properties"],
                            values=torch.arange(
                                node_features.shape[-1], device=node_features.device
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

        atomic_features_dict: Dict[str, torch.Tensor] = {}
        for output_name, head in self.heads.items():
            atomic_features_dict[output_name] = head(node_features)

        # output the last-layer features for the outputs, if requested:
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
                f"mtt::{base_name}" not in atomic_features_dict
                and base_name not in atomic_features_dict
            ):
                raise ValueError(
                    f"Features {output_name} can only be requested "
                    f"if the corresponding output {base_name} is also requested."
                )
            if f"mtt::{base_name}" in atomic_features_dict:
                base_name = f"mtt::{base_name}"
            last_layer_feature_tmap = TensorMap(
                keys=self.single_label,
                blocks=[
                    TensorBlock(
                        values=atomic_features_dict[base_name],
                        samples=sample_labels,
                        components=[],
                        properties=Labels(
                            names=["properties"],
                            values=torch.arange(
                                atomic_features_dict[base_name].shape[-1],
                                device=atomic_features_dict[base_name].device,
                            ).reshape(-1, 1),
                        ),
                    )
                ],
            )
            last_layer_features_options = outputs[output_name]
            if last_layer_features_options.per_atom:
                return_dict[output_name] = last_layer_feature_tmap
            else:
                return_dict[output_name] = sum_over_atoms(
                    last_layer_feature_tmap,
                )

        atomic_properties_tmap_dict: Dict[str, TensorMap] = {}
        for output_name, last_layer in self.last_layers.items():
            if output_name in outputs:
                atomic_features = atomic_features_dict[output_name]
                atomic_properties_by_block = []
                for last_layer_by_block in last_layer.values():
                    atomic_properties_by_block.append(
                        last_layer_by_block(atomic_features)
                    )
                blocks = [
                    TensorBlock(
                        values=atomic_property.reshape([-1] + shape),
                        samples=sample_labels,
                        components=components,
                        properties=properties,
                    )
                    for atomic_property, shape, components, properties in zip(
                        atomic_properties_by_block,
                        self.output_shapes[output_name].values(),
                        self.component_labels[output_name],
                        self.property_labels[output_name],
                    )
                ]
                atomic_properties_tmap_dict[output_name] = TensorMap(
                    keys=self.key_labels[output_name],
                    blocks=blocks,
                )

        if selected_atoms is not None:
            for output_name, tmap in atomic_properties_tmap_dict.items():
                atomic_properties_tmap_dict[output_name] = metatensor.torch.slice(
                    tmap, axis="samples", selection=selected_atoms
                )

        for output_name, atomic_property in atomic_properties_tmap_dict.items():
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
                    return_dict[name] = metatensor.torch.add(
                        return_dict[name],
                        additive_contributions[name],
                    )

        return return_dict

    def requested_neighbor_lists(
        self,
    ) -> List[NeighborListOptions]:
        return [self.requested_nl]

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path]) -> "NanoPET":
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
        model.additive_models[0].sync_tensor_maps()

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

        # Additionally, the composition model contains some `TensorMap`s that cannot
        # be registered correctly with Pytorch. This funciton moves them:
        self.additive_models[0]._move_weights_to_device_and_dtype(
            torch.device("cpu"), torch.float64
        )

        interaction_ranges = [self.hypers["num_gnn_layers"] * self.hypers["cutoff"]]
        for additive_model in self.additive_models:
            if hasattr(additive_model, "cutoff_radius"):
                interaction_ranges.append(additive_model.cutoff_radius)
            if self.long_range:
                interaction_ranges.append(torch.inf)
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

    def _add_output(self, target_name: str, target_info: TargetInfo) -> None:
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
        if (
            target_name not in self.head_types  # default to MLP
            or self.head_types[target_name] == "mlp"
        ):
            self.heads[target_name] = torch.nn.Sequential(
                torch.nn.Linear(
                    self.hypers["d_pet"], 4 * self.hypers["d_pet"], bias=False
                ),
                torch.nn.SiLU(),
                torch.nn.Linear(
                    4 * self.hypers["d_pet"], self.hypers["d_pet"], bias=False
                ),
                torch.nn.SiLU(),
            )
        elif self.head_types[target_name] == "linear":
            self.heads[target_name] = torch.nn.Sequential()
        else:
            raise ValueError(
                f"Unsupported head type {self.head_types[target_name]} "
                f"for target {target_name}"
            )

        ll_features_name = (
            f"mtt::aux::{target_name.replace('mtt::', '')}_last_layer_features"
        )
        self.outputs[ll_features_name] = ModelOutput(per_atom=True)

        self.last_layers[target_name] = torch.nn.ModuleDict(
            {
                key: torch.nn.Linear(
                    self.hypers["d_pet"],
                    prod(shape),
                    bias=False,
                )
                for key, shape in self.output_shapes[target_name].items()
            }
        )

        self.key_labels[target_name] = target_info.layout.keys
        self.component_labels[target_name] = [
            block.components for block in target_info.layout.blocks()
        ]
        self.property_labels[target_name] = [
            block.properties for block in target_info.layout.blocks()
        ]


def manual_prod(shape: List[int]) -> int:
    # prod from standard library not supported in torchscript
    result = 1
    for dim in shape:
        result *= dim
    return result
