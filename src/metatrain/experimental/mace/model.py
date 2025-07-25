import warnings
from math import prod
from typing import Any, Dict, List, Literal, Optional

import metatensor.torch
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
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

from mace.modules import MACE, RealAgnosticResidualInteractionBlock
from e3nn import o3

from .utils.structures import create_batch

@torch.jit.interface
class LinearInterface(torch.nn.Module):

    def forward(self, features, weight: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass

from typing import TypedDict

class Target(TypedDict):
    is_cartesian: bool

class MetaMACE(ModelInterface):
    """Interface of MACE for metatrain."""

    __checkpoint_version__ = 1
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float64, torch.float32]
    __default_metadata__ = ModelMetadata(
        # references={"architecture": ["https://arxiv.org/abs/2305.19302v3"]}
    )

    component_labels: Dict[str, List[List[Labels]]]

    def __init__(self, model_hypers: Dict, dataset_info: DatasetInfo) -> None:
        super().__init__(model_hypers, dataset_info)
        # checks on targets inside the RotationalAugmenter class in the trainer

        self.dataset_info = dataset_info
        self.new_outputs = list(dataset_info.targets.keys())
        self.atomic_types = dataset_info.atomic_types
        self.register_buffer(
            "atomic_types_to_species_index", torch.zeros(max(dataset_info.atomic_types) + 1, dtype=torch.int64)
        )
        for i, atomic_type in enumerate(dataset_info.atomic_types):
            self.atomic_types_to_species_index[atomic_type] = i

        self.requested_nl = NeighborListOptions(
            cutoff=self.hypers["cutoff"],
            full_list=True,
            strict=True,
        )

        self.cutoff = float(self.hypers["cutoff"])
        self.cutoff_width = float(self.hypers["cutoff_width"])

        self.mace_model = MACE(
            r_max=self.cutoff,
            num_bessel=model_hypers["num_bessel"],
            num_polynomial_cutoff=model_hypers["num_polynomial_cutoff"],
            max_ell=model_hypers["max_ell"],
            interaction_cls= RealAgnosticResidualInteractionBlock,
            interaction_cls_first=RealAgnosticResidualInteractionBlock,
            num_interactions=model_hypers["num_interactions"],
            num_elements=len(dataset_info.atomic_types),
            hidden_irreps=o3.Irreps(model_hypers["hidden_irreps"]),
            MLP_irreps=o3.Irreps(model_hypers["MLP_irreps"]),
            atomic_energies=torch.zeros(len(dataset_info.atomic_types)),
            avg_num_neighbors=model_hypers["avg_num_neighbors"],
            atomic_numbers=torch.arange(len(dataset_info.atomic_types)),
            correlation=model_hypers["correlation"],
            gate=model_hypers.get("gate", None),
        )

        self.outputs = {
            "features": ModelOutput(unit="", per_atom=True)
        }
        self.output_shapes: Dict[str, Dict[str, List[int]]] = {}
        self.key_labels: Dict[str, Labels] = {}
        self.component_labels: Dict[str, List[List[Labels]]] = {}
        self.property_labels: Dict[str, List[Labels]] = {}
        self.heads: Dict[str, torch.nn.Module] = torch.nn.ModuleDict()
        self.target_infos: Dict[str, TargetInfo] = {}
        for target_name, target_info in dataset_info.targets.items():
            self._add_output(target_name, target_info)

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
        # if self.hypers["zbl"]:
        #     additive_models.append(
        #         ZBL(
        #             {},
        #             dataset_info=DatasetInfo(
        #                 length_unit=dataset_info.length_unit,
        #                 atomic_types=self.atomic_types,
        #                 targets={
        #                     target_name: target_info
        #                     for target_name, target_info in dataset_info.targets.items()
        #                     if ZBL.is_valid_target(target_name, target_info)
        #                 },
        #             ),
        #         )
        #     )
        self.additive_models = torch.nn.ModuleList(additive_models)

        self.scaler = Scaler(hypers={}, dataset_info=dataset_info)

        self.single_label = Labels.single()

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs

    def restart(self, dataset_info: DatasetInfo) -> MACE:
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
        
        if selected_atoms is not None:
            raise NotImplementedError(
                "selected_atoms is not supported in MetaMACE for now. "
            )
        
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

        data = create_batch(
            systems=systems,
            neighbor_list_options=self.requested_nl,
            atomic_types_to_species_index=self.atomic_types_to_species_index,
            n_types=len(self.atomic_types),
        )

        # Change coordinates to YZX
        data["positions"] = data["positions"][:, [1, 2, 0]]

        mace_output = self.mace_model(data, training=self.training)

        sample_labels = Labels(
            names=["system"],
            values=torch.arange(len(systems), device=device).reshape(-1, 1),
        )

        return_dict: Dict[str, TensorMap] = {}

        # return_dict["energy"] = TensorMap(
        #     keys=Labels.single(),
        #     blocks=[
        #         TensorBlock(
        #             values=mace_output["energy"].reshape(-1, 1),
        #             samples=sample_labels,
        #             components=[],
        #             properties=Labels(
        #                 names=["energy"],
        #                 values=torch.tensor([[0]], device=device),
        #             ),
        #         )
        #     ]
        # )

        # output the hidden features, if requested:
        # if "features" in outputs:
        #     feature_tmap = TensorMap(
        #         keys=self.single_label,
        #         blocks=[
        #             TensorBlock(
        #                 values=node_features,
        #                 samples=sample_labels,
        #                 components=[],
        #                 properties=Labels(
        #                     names=["properties"],
        #                     values=torch.arange(
        #                         node_features.shape[-1], device=node_features.device
        #                     ).reshape(-1, 1),
        #                 ),
        #             )
        #         ],
        #     )
        #     features_options = outputs["features"]
        #     if features_options.per_atom:
        #         return_dict["features"] = feature_tmap
        #     else:
        #         return_dict["features"] = sum_over_atoms(feature_tmap)

        # atomic_features_dict: Dict[str, torch.Tensor] = {}
        # for output_name, head in self.heads.items():
        #     atomic_features_dict[output_name] = head(node_features)

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
        atom_sample_labels = Labels(
            names=["system", "atom"],
            values=sample_values,
        )

        node_features = mace_output["node_feats"]
        assert node_features is not None, "Node features should not be None"
        # # output the last-layer features for the outputs, if requested:
        for output_name in outputs.keys():
            head: LinearInterface = self.heads[output_name]
            node_target = head.forward(node_features, weight=None, bias=None)
            is_cartesian = self.target_infos[output_name]["is_cartesian"]

            blocks: list[TensorBlock] = []
            pointer = 0
            for i in range(len(self.component_labels[output_name])):

                components = self.component_labels[output_name][i]
                properties = self.property_labels[output_name][i]

                has_components = len(components) > 0
                n_components = len(components[0]) if has_components else 1
                n_properties = len(properties)

                end = pointer + n_components * n_properties

                values = node_target[:, pointer:end].reshape(
                    -1, n_properties, n_components,
                ).transpose(1, 2)

                if is_cartesian and n_components == 3:
                    # Go back from YZX to XYZ
                    values = values[:, [2, 0, 1], :]

                if not has_components:
                    # Remove the components dimension if there are no components
                    values = values.squeeze(1)

                blocks.append(
                    TensorBlock(
                        values=values,
                        samples=atom_sample_labels,
                        components=components,
                        properties=properties,
                    )
                )
                pointer = end

            atom_target = TensorMap(
                keys=self.key_labels[output_name],
                blocks=blocks
            )

            if outputs[output_name].per_atom:
                return_dict[output_name] = atom_target
            else:
                return_dict[output_name] = sum_over_atoms(atom_target)

            # last_layer_feature_tmap = TensorMap(
            #     keys=self.single_label,
            #     blocks=[
            #         TensorBlock(
            #             values=atomic_features_dict[base_name],
            #             samples=sample_labels,
            #             components=[],
            #             properties=Labels(
            #                 names=["properties"],
            #                 values=torch.arange(
            #                     atomic_features_dict[base_name].shape[-1],
            #                     device=atomic_features_dict[base_name].device,
            #                 ).reshape(-1, 1),
            #             ),
            #         )
            #     ],
            # )
            # last_layer_features_options = outputs[output_name]
            # if last_layer_features_options.per_atom:
            #     return_dict[output_name] = last_layer_feature_tmap
            # else:
            #     return_dict[output_name] = sum_over_atoms(
            #         last_layer_feature_tmap,
            #     )

        # atomic_properties_tmap_dict: Dict[str, TensorMap] = {}
        # for output_name, last_layer in self.last_layers.items():
        #     if output_name in outputs:
        #         atomic_features = atomic_features_dict[output_name]
        #         atomic_properties_by_block = []
        #         for last_layer_by_block in last_layer.values():
        #             atomic_properties_by_block.append(
        #                 last_layer_by_block(atomic_features)
        #             )
        #         all_components = self.component_labels[output_name]
        #         if len(all_components[0]) == 2 and all(
        #             "xyz" in comp.names[0] for comp in all_components[0]
        #         ):
        #             # rank-2 Cartesian tensor, symmetrize
        #             tensor_as_three_by_three = atomic_properties_by_block[0].reshape(
        #                 -1, 3, 3, list(self.output_shapes[output_name].values())[0][-1]
        #             )
        #             volumes = torch.stack(
        #                 [torch.abs(torch.det(system.cell)) for system in systems]
        #             )
        #             volumes_by_atom = (
        #                 volumes[system_indices].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        #             )
        #             tensor_as_three_by_three = (
        #                 tensor_as_three_by_three / volumes_by_atom
        #             )
        #             tensor_as_three_by_three = (
        #                 tensor_as_three_by_three
        #                 + tensor_as_three_by_three.transpose(1, 2)
        #             ) / 2.0
        #             atomic_properties_by_block[0] = tensor_as_three_by_three

        #         blocks = [
        #             TensorBlock(
        #                 values=atomic_property.reshape([-1] + shape),
        #                 samples=sample_labels,
        #                 components=components,
        #                 properties=properties,
        #             )
        #             for atomic_property, shape, components, properties in zip(
        #                 atomic_properties_by_block,
        #                 self.output_shapes[output_name].values(),
        #                 self.component_labels[output_name],
        #                 self.property_labels[output_name],
        #             )
        #         ]
        #         atomic_properties_tmap_dict[output_name] = TensorMap(
        #             keys=self.key_labels[output_name],
        #             blocks=blocks,
        #         )

        # if selected_atoms is not None:
        #     for output_name, tmap in atomic_properties_tmap_dict.items():
        #         atomic_properties_tmap_dict[output_name] = metatensor.torch.slice(
        #             tmap, axis="samples", selection=selected_atoms
        #         )

        # for output_name, atomic_property in atomic_properties_tmap_dict.items():
        #     if outputs[output_name].per_atom:
        #         return_dict[output_name] = atomic_property
        #     else:
        #         return_dict[output_name] = sum_over_atoms(atomic_property)

        # if not self.training:
        #     # at evaluation, we also introduce the scaler and additive contributions
        #     return_dict = self.scaler(return_dict)
        #     for additive_model in self.additive_models:
        #         outputs_for_additive_model: Dict[str, ModelOutput] = {}
        #         for name, output in outputs.items():
        #             if name in additive_model.outputs:
        #                 outputs_for_additive_model[name] = output
        #         additive_contributions = additive_model(
        #             systems,
        #             outputs_for_additive_model,
        #             selected_atoms,
        #         )
        #         for name in additive_contributions:
        #             return_dict[name] = metatensor.torch.add(
        #                 return_dict[name],
        #                 additive_contributions[name],
        #             )

        return return_dict

    def requested_neighbor_lists(
        self,
    ) -> List[NeighborListOptions]:
        return [self.requested_nl]

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> "NanoPET":
        model_data = checkpoint["model_data"]

        if context == "restart":
            model_state_dict = checkpoint["model_state_dict"]
        elif context == "finetune" or context == "export":
            model_state_dict = checkpoint["best_model_state_dict"]
        else:
            raise ValueError("Unknown context tag for checkpoint loading!")

        # Create the model
        model = cls(**model_data)
        state_dict_iter = iter(model_state_dict.values())
        next(state_dict_iter)  # skip `species_to_species_index` buffer (int)
        dtype = next(state_dict_iter).dtype
        model.to(dtype).load_state_dict(model_state_dict)

        # Loading the metadata from the checkpoint
        metadata = checkpoint.get("metadata", None)
        if metadata is not None:
            model.__default_metadata__ = metadata

        return model

    def export(self, metadata: Optional[ModelMetadata] = None) -> AtomisticModel:
        dtype = next(self.parameters()).dtype
        if dtype not in self.__supported_dtypes__:
            raise ValueError(f"unsupported dtype {dtype} for NanoPET")

        # Make sure the model is all in the same dtype
        # For example, after training, the additive models could still be in
        # float64
        self.to(dtype)

        interaction_ranges = [self.hypers["num_interactions"] * self.hypers["cutoff"]]
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
            if len(target_info.layout.block().components) > 1:
                raise ValueError(
                    "MetaMACE does not support Cartesian tensors with rank > 1."
                )

        # one output shape for each tensor block, grouped by target (i.e. tensormap)
        self.output_shapes[target_name] = {}
        irreps = []
        for key, block in target_info.layout.items():
            dict_key = target_name
            multiplicity = len(block.properties.values)

            if target_info.is_scalar:
                dict_key += f"_{key.names[0]}_{int(key.values[0])}"
                irreps.append((multiplicity, (0, 1)))
            elif target_info.is_spherical:
                l = int(key["o3_lambda"])
                dict_key += f"_o3_lambda_{l}"
                irreps.append((multiplicity, (l, (-1)**l)))                  
            elif target_info.is_cartesian:
                l = 1
                irreps.append((multiplicity, (l, (-1)**l)))
            
            self.output_shapes[target_name][dict_key] = [
                len(comp.values) for comp in block.components
            ] + [len(block.properties.values)]

        self.outputs[target_name] = ModelOutput(
            quantity=target_info.quantity,
            unit=target_info.unit,
            per_atom=True,
        )

        hidden_irreps = o3.Irreps(self.hypers["hidden_irreps"])
        n_scalars = hidden_irreps.count((0, 1)) 
        mace_out_irreps = hidden_irreps * (self.hypers["num_interactions"] - 1) + o3.Irreps([(n_scalars, (0, 1))])

        self.heads[target_name] = o3.Linear(
            irreps_in=mace_out_irreps,
            irreps_out=o3.Irreps(irreps)
        )

        self.heads[target_name].to(torch.float64)

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
        self.target_infos[target_name] = target_info #Target(is_cartesian=target_info.is_cartesian)
    
    @staticmethod
    def upgrade_checkpoint(checkpoint: Dict) -> Dict:
        raise NotImplementedError("checkpoint upgrade is not implemented for MetaMACE")


def manual_prod(shape: List[int]) -> int:
    # prod from standard library not supported in torchscript
    result = 1
    for dim in shape:
        result *= dim
    return result
