import logging
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
from ...utils.metadata import append_metadata_references
from ...utils.scaler import Scaler
from .modules.heads import (
    Head,
)
from .modules.nef import (
    compute_neighbors_pos,
    edge_array_to_nef,
    get_corresponding_edges,
    get_nef_indices,
)
from .modules.structures import concatenate_structures
from .modules.transformer import CartesianTransformer
from .modules.utilities import cutoff_func


logger = logging.getLogger(__name__)


class NativePET(torch.nn.Module):
    """
    Native metatrain implementation of the PET architecture.

    Originally proposed in work (https://arxiv.org/abs/2305.19302v3),
    and published in the `pet` package (https://github.com/spozdn/pet).

    """

    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float32]
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
        self.residual_factor = float(self.hypers["residual_factor"])
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

        self.heads = torch.nn.ModuleDict()
        self.bond_heads = torch.nn.ModuleDict()
        self.last_layers = torch.nn.ModuleDict()
        self.bond_last_layers = torch.nn.ModuleDict()
        # last-layer feature size (for LLPR module)
        self.last_layer_feature_size = (
            self.hypers["num_gnn_layers"] * self.hypers["d_head"] * 2
        )
        # if they are enabled, the edge features are concatenated
        # to the node features

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
            torch.full(
                (max(self.atomic_types) + 1,),
                -1,
            ),
        )
        for i, species in enumerate(self.atomic_types):
            self.species_to_species_index[species] = i

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

    def restart(self, dataset_info: DatasetInfo) -> "NativePET":
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
                "The NativePET model does not support adding new atomic types."
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

        nl_options = self.requested_neighbor_lists()[0]

        (
            positions,
            centers,
            neighbors,
            species,
            cells,
            cell_shifts,
        ) = concatenate_structures(systems, nl_options)

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

        # Element indices
        element_indices_nodes = self.species_to_species_index[species]
        element_indices_neighbors = element_indices_nodes[neighbors]

        # Send everything to NEF:
        edge_vectors = edge_array_to_nef(edge_vectors, nef_indices)
        element_indices_neighbors = edge_array_to_nef(
            element_indices_neighbors, nef_indices
        )

        corresponding_edges = get_corresponding_edges(
            torch.concatenate(
                [centers.unsqueeze(-1), neighbors.unsqueeze(-1), cell_shifts],
                dim=-1,
            )
        )
        neighbors_index = edge_array_to_nef(neighbors, nef_indices)
        neighbors_pos = compute_neighbors_pos(
            nef_indices, corresponding_edges, nef_mask
        )

        mask = torch.logical_not(nef_mask)

        lengths = torch.sqrt(torch.sum(edge_vectors * edge_vectors, dim=2) + 1e-16)
        multipliers = cutoff_func(lengths, self.cutoff, self.cutoff_width)
        multipliers[mask] = 0.0

        input_messages = self.embedding(element_indices_neighbors)

        return_dict: Dict[str, TensorMap] = {}
        central_tokens_list = []
        output_messages_list = []

        for gnn_layer in self.gnn_layers:
            result = gnn_layer(
                edge_vectors,
                element_indices_nodes,
                element_indices_neighbors,
                input_messages,
                mask,
                bincount,
            )
            output_messages = result["output_messages"]
            new_input_messages = output_messages[neighbors_index, neighbors_pos]
            input_messages = self.residual_factor * (
                input_messages + new_input_messages
            )
            central_tokens_list.append(result["central_token"])
            output_messages_list.append(output_messages)

        central_tokens_features = torch.cat(central_tokens_list, dim=1)
        output_messages_features = torch.cat(output_messages_list, dim=2)
        output_messages_features = output_messages_features * multipliers[:, :, None]
        output_messages_features = output_messages_features.sum(dim=1)
        features = torch.cat([central_tokens_features, output_messages_features], dim=1)

        # output the hidden features, if requested:
        if "features" in outputs:
            feature_tmap = TensorMap(
                keys=self.single_label,
                blocks=[
                    TensorBlock(
                        values=features,
                        samples=sample_labels,
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
                return_dict["features"] = metatensor.torch.sum_over_samples(
                    feature_tmap, ["atom"]
                )

        central_tokens_features_dict: Dict[str, List[torch.Tensor]] = {}
        messages_bonds_features_dict: Dict[str, List[torch.Tensor]] = {}

        for output_name, heads in self.heads.items():
            if output_name not in central_tokens_features_dict:
                central_tokens_features_dict[output_name] = []
            for i, head in enumerate(heads):
                central_tokens = central_tokens_list[i]
                central_tokens_feature = head(central_tokens)
                central_tokens_features_dict[output_name].append(central_tokens_feature)

        for output_name, bond_heads in self.bond_heads.items():
            if output_name not in messages_bonds_features_dict:
                messages_bonds_features_dict[output_name] = []
            for i, bond_head in enumerate(bond_heads):
                output_messages = output_messages_list[i]
                messages_bond_feature = bond_head(output_messages)
                messages_bonds_features_dict[output_name].append(messages_bond_feature)

        last_layer_features_dict: Dict[str, List[torch.Tensor]] = {}
        last_layer_features: Dict[str, torch.Tensor] = {}

        for output_name in self.target_names:
            if output_name not in last_layer_features_dict:
                last_layer_features_dict[output_name] = []
            for i in range(len(central_tokens_features_dict[output_name])):
                central_tokens_feature = central_tokens_features_dict[output_name][i]
                messages_bond_feature = messages_bonds_features_dict[output_name][i]
                messages_bond_feature = messages_bond_feature * multipliers[:, :, None]
                messages_bond_feature = messages_bond_feature.sum(dim=1)
                last_layer_features_dict[output_name].append(central_tokens_feature)
                last_layer_features_dict[output_name].append(messages_bond_feature)
            last_layer_features[output_name] = torch.cat(
                last_layer_features_dict[output_name], dim=1
            )

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
                f"mtt::{base_name}" not in last_layer_features
                and base_name not in last_layer_features
            ):
                raise ValueError(
                    f"Features {output_name} can only be requested "
                    f"if the corresponding output {base_name} is also requested."
                )
            if f"mtt::{base_name}" in last_layer_features:
                base_name = f"mtt::{base_name}"
            last_layer_feature_tmap = TensorMap(
                keys=self.single_label,
                blocks=[
                    TensorBlock(
                        values=last_layer_features[base_name],
                        samples=sample_labels,
                        components=[],
                        properties=Labels(
                            names=["properties"],
                            values=torch.arange(
                                last_layer_features[base_name].shape[-1],
                                device=last_layer_features[base_name].device,
                            ).reshape(-1, 1),
                        ),
                    )
                ],
            )
            last_layer_features_options = outputs[output_name]
            if last_layer_features_options.per_atom:
                return_dict[output_name] = last_layer_feature_tmap
            else:
                return_dict[output_name] = metatensor.torch.sum_over_samples(
                    last_layer_feature_tmap, ["atom"]
                )

        atomic_properties_tmap_dict: Dict[str, TensorMap] = {}

        central_tokens_properties_by_layer: List[List[torch.Tensor]] = []
        messages_bonds_properties_by_layer: List[List[torch.Tensor]] = []

        for output_name, last_layers in self.last_layers.items():
            if output_name in outputs:
                for i, last_layer in enumerate(last_layers):
                    central_tokens_features = central_tokens_features_dict[output_name][
                        i
                    ]
                    central_tokens_properties_by_block: List[torch.Tensor] = []
                    for last_layer_by_block in last_layer.values():
                        central_tokens_properties_by_block.append(
                            last_layer_by_block(central_tokens_features)
                        )

                    central_tokens_properties_by_layer.append(
                        central_tokens_properties_by_block
                    )

        for output_name, last_layers in self.bond_last_layers.items():
            if output_name in outputs:
                for i, last_layer in enumerate(last_layers):
                    messages_bonds_features = messages_bonds_features_dict[output_name][
                        i
                    ]
                    messages_bonds_properties_by_block: List[torch.Tensor] = []
                    for last_layer_by_block in last_layer.values():
                        messages_bonds_properties = last_layer_by_block(
                            messages_bonds_features
                        )
                        mask_expanded = mask[..., None].repeat(
                            1, 1, messages_bonds_properties.shape[2]
                        )
                        messages_bonds_properties = torch.where(
                            mask_expanded, 0.0, messages_bonds_properties
                        )
                        messages_bonds_properties = (
                            messages_bonds_properties * multipliers[:, :, None]
                        )
                        messages_bonds_properties_by_block.append(
                            messages_bonds_properties.sum(dim=1)
                        )
                    messages_bonds_properties_by_layer.append(
                        messages_bonds_properties_by_block
                    )

        for output_name in self.target_names:
            if output_name in outputs:
                atomic_properties_by_block = {
                    key: torch.zeros(
                        1, dtype=edge_vectors.dtype, device=edge_vectors.device
                    )
                    for key in self.output_shapes[output_name].keys()
                }

                for i in range(len(central_tokens_properties_by_layer)):
                    central_tokens_properties_by_block = (
                        central_tokens_properties_by_layer[i]
                    )
                    messages_bonds_properties_by_block = (
                        messages_bonds_properties_by_layer[i]
                    )
                    for j, key in enumerate(atomic_properties_by_block):
                        central_tokens_properties = central_tokens_properties_by_block[
                            j
                        ]
                        messages_bonds_properties = messages_bonds_properties_by_block[
                            j
                        ]
                        atomic_properties_by_block[key] = atomic_properties_by_block[
                            key
                        ] + (central_tokens_properties + messages_bonds_properties)

                blocks = [
                    TensorBlock(
                        values=atomic_properties_by_block[key].reshape([-1] + shape),
                        samples=sample_labels,
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
                return_dict[output_name] = metatensor.torch.sum_over_samples(
                    atomic_property, ["atom"]
                )

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

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path]) -> "NativePET":
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
            raise ValueError(f"unsupported dtype {dtype} for NativePET")

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

        self.heads[target_name] = torch.nn.ModuleList(
            [
                Head(self.hypers["d_pet"], self.hypers["d_head"])
                for _ in range(self.hypers["num_gnn_layers"])
            ]
        )

        self.bond_heads[target_name] = torch.nn.ModuleList(
            [
                Head(self.hypers["d_pet"], self.hypers["d_head"])
                for _ in range(self.hypers["num_gnn_layers"])
            ]
        )

        self.last_layers[target_name] = torch.nn.ModuleList(
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

        self.bond_last_layers[target_name] = torch.nn.ModuleList(
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
