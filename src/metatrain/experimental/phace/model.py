import logging
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple

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
from metatrain.utils.data.dataset import DatasetInfo, TargetInfo
from metatrain.utils.dtype import dtype_to_str
from metatrain.utils.metadata import merge_metadata
from metatrain.utils.scaler import Scaler

from metatrain.experimental.phace.documentation import ModelHypers
from metatrain.experimental.phace.modules.base_model import GradientModel
from metatrain.experimental.phace.modules.layers import Linear
from metatrain.experimental.phace.utils import systems_to_list


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=("The TorchScript type system doesn't support instance-level annotations"),
)


class PhACE(ModelInterface[ModelHypers]):
    __checkpoint_version__ = 1
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float64, torch.float32]
    __default_metadata__ = ModelMetadata(references={})

    component_labels: Dict[str, List[List[Labels]]]
    U_dict: Dict[int, torch.Tensor]
    # U_dict_parity: Dict[str, torch.Tensor]

    def __init__(self, hypers: Dict, dataset_info: DatasetInfo) -> None:
        super().__init__(hypers, dataset_info, self.__default_metadata__)

        self.hypers = hypers
        self.dataset_info = dataset_info
        self.new_outputs = list(dataset_info.targets.keys())
        self.atomic_types = sorted(dataset_info.atomic_types)

        self.cutoff_radius = hypers["cutoff"]
        self.dataset_info = dataset_info
        self.hypers = hypers

        self.module = GradientModel(hypers, self.atomic_types)
        self.k_max_l = self.module.k_max_l

        self.overall_scaling = hypers["overall_scaling"]

        self.outputs = {
            "features": ModelOutput(unit="", per_atom=True)
        }  # the model is always capable of outputting the internal features
        for target_name in dataset_info.targets.keys():
            # the model can always output the last-layer features for the targets
            ll_features_name = (
                f"mtt::aux::{target_name.replace('mtt::', '')}_last_layer_features"
            )
            self.outputs[ll_features_name] = ModelOutput(per_atom=True)

        self.requested_LS_tuples: List[Tuple[int, int]] = []
        self.heads = torch.nn.ModuleDict()
        self.head_types = self.hypers["heads"]
        self.last_layers = torch.nn.ModuleDict()
        self.key_labels: Dict[str, Labels] = {}
        self.component_labels: Dict[str, List[List[Labels]]] = {}
        self.property_labels: Dict[str, List[Labels]] = {}
        self.head_num_layers = self.hypers["head_num_layers"]
        for target_name, target_info in dataset_info.targets.items():
            self._add_output(target_name, target_info)

        self.last_layer_feature_size = self.k_max_l[0]

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
        if self.hypers["zbl"]:
            additive_models.append(ZBL(hypers, dataset_info))
        self.additive_models = torch.nn.ModuleList(additive_models)

        # scaler: this is also handled by the trainer at training time
        self.scaler = Scaler(hypers={}, dataset_info=dataset_info)

        self.single_label = Labels.single()

    @torch.jit.export
    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs

    def restart(self, dataset_info: DatasetInfo) -> "PhACE":
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
                "The PhACE model does not support adding new atomic types."
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
        # transfer labels, if needed
        device = systems[0].device
        if self.single_label.values.device != device:
            self.single_label = self.single_label.to(device)
            self.key_labels = {
                output_name: label.to(device)
                for output_name, label in self.key_labels.items()
            }
            self.component_labels = {
                output_name: [
                    [label.to(device) for label in component]
                    for component in components
                ]
                for output_name, components in self.component_labels.items()
            }
            self.property_labels = {
                output_name: [label.to(device) for label in labels]
                for output_name, labels in self.property_labels.items()
            }

        # compute sample labels
        centers_list = []
        structures_centers_list = []
        for i, system in enumerate(systems):
            centers_list.append(
                torch.arange(len(system.positions), device=device, dtype=torch.int32)
            )
            structures_centers_list.append(
                torch.tensor([i] * len(system.positions), device=device, dtype=torch.int32)
            )
        centers = torch.cat(centers_list, dim=0)
        structure_centers = torch.cat(structures_centers_list, dim=0)
        samples_values = torch.stack([structure_centers, centers], dim=1)
        samples = metatensor.torch.Labels(
            names=["system", "atom"],
            values=samples_values,
        )

        neighbor_list_options = self.requested_neighbor_lists()[0]  # there is only one
        structures_as_list = systems_to_list(systems, neighbor_list_options)
        energies, forces = self.module(structures_as_list)

        # TODO: selected_atoms

        # features = TensorMap(
        #     keys=spherical_expansion.keys,
        #     blocks=[
        #         TensorBlock(
        #             values=embedded_features[l],
        #             samples=spherical_expansion.block({"o3_lambda": l}).samples,
        #             components=spherical_expansion.block({"o3_lambda": l}).components,
        #             properties=spherical_expansion.block({"o3_lambda": l}).properties,
        #         )
        #         for l in range(self.l_max + 1)
        #     ],
        # )

        # # remove the center_type dimension
        # features = metatensor.torch.remove_dimension(features, "samples", "center_type")

        # if selected_atoms is not None:
        #     features = metatensor.torch.slice(
        #         features, axis="samples", selection=selected_atoms
        #     )

        return_dict: Dict[str, TensorMap] = {}

        # # output the hidden features, if requested (invariant only):
        # if "features" in outputs:
        #     feature_tmap = TensorMap(
        #         keys=self.single_label,
        #         blocks=[
        #             TensorBlock(
        #                 values=features.block(
        #                     {"o3_lambda": 0, "o3_sigma": 1}
        #                 ).values.squeeze(1),
        #                 samples=features.block({"o3_lambda": 0, "o3_sigma": 1}).samples,
        #                 components=[],
        #                 properties=features.block(
        #                     {"o3_lambda": 0, "o3_sigma": 1}
        #                 ).properties,
        #             )
        #         ],
        #     )
        #     features_options = outputs["features"]
        #     if features_options.per_atom:
        #         return_dict["features"] = feature_tmap
        #     else:
        #         return_dict["features"] = metatensor.torch.sum_over_samples(
        #             feature_tmap, ["atom"]
        #         )

        for output_name, output_head in self.heads.items():
            if output_name in outputs:
                block = TensorBlock(
                    values=energies.squeeze(1),
                    samples=samples,
                    components=self.component_labels[output_name][0],
                    properties=self.property_labels[output_name][0],
                )
                block = metatensor.torch.sum_over_samples_block(block, ["atom"])
                block.add_gradient(
                    "positions",
                    TensorBlock(
                        values=-forces.unsqueeze(-1),
                        samples = Labels(
                            names=["sample", "atom"],
                            values=torch.stack(
                                [
                                    torch.concatenate(
                                        [
                                            torch.tensor([i] * len(system), device=device)
                                            for i, system in enumerate(systems)
                                        ]
                                    ),
                                    torch.concatenate(
                                        [torch.arange(len(system), device=device) for system in systems]
                                    ),
                                ],
                                dim=1,
                            ),
                            assume_unique=True,
                        ),
                        components=[
                            Labels(
                                names=["xyz"],
                                values=torch.tensor([[0], [1], [2]], device=device),
                            )
                        ],
                        properties=self.property_labels[output_name][0],
                    )
                )
                return_dict[output_name] = TensorMap(
                    keys=self.key_labels[output_name],
                    blocks=[block],
                )

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
            if f"mtt::{base_name}" not in return_dict and base_name not in return_dict:
                raise ValueError(
                    f"Features {output_name} can only be requested "
                    f"if the corresponding output {base_name} is also requested."
                )
            if f"mtt::{base_name}" in return_dict:
                base_name = f"mtt::{base_name}"
            return_dict[output_name] = return_dict[base_name]
            last_layer_features_options = outputs[output_name]
            if not last_layer_features_options.per_atom:
                return_dict[output_name] = metatensor.torch.sum_over_samples(
                    return_dict[output_name], ["atom"]
                )

        for output_name, output_layer in self.last_layers.items():
            if output_name in outputs:
                return_dict[output_name] = metatensor.torch.multiply(
                    return_dict[output_name],
                    self.overall_scaling,
                )
            if len(self.component_labels[output_name][0]) > 0:
                if self.component_labels[output_name][0][0].names == ["xyz"]:
                    # modify to extract xyz from spherical L=1
                    tmap_as_spherical = return_dict[output_name]
                    cartesian_values = tmap_as_spherical.block().values[:, [2, 0, 1]]
                    return_dict[output_name] = TensorMap(
                        keys=self.key_labels[output_name],
                        blocks=[
                            TensorBlock(
                                values=cartesian_values,
                                samples=tmap_as_spherical.block().samples,
                                components=self.component_labels[output_name][0],
                                properties=self.property_labels[output_name][0],
                            )
                        ],
                    )

        # for output_name in self.last_layers:
        #     if output_name in outputs:
        #         if not outputs[output_name].per_atom:
        #             return_dict[output_name] = metatensor.torch.sum_over_samples(
        #                 return_dict[output_name], ["atom"]
        #             )

        if not self.training:
            # at evaluation, we also introduce the scaler and additive contributions
            return_dict = self.scaler(systems, return_dict)
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
                    # TODO: uncomment this after metatensor.torch.add
                    # is updated to handle sparse sums
                    # return_dict[name] = metatensor.torch.add(
                    #     return_dict[name],
                    #     additive_contributions[name].to(
                    #         device=return_dict[name].device,
                    #         dtype=return_dict[name].dtype
                    #         ),
                    # )
                    # TODO: "manual" sparse sum: update to metatensor.torch.add
                    # after sparse sum is implemented in metatensor.operations
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

    def requested_neighbor_lists(
        self,
    ) -> List[NeighborListOptions]:
        return [self.requested_nl]

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> "PhACE":
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
        state_dict_iterator = iter(model_state_dict.values())
        next(state_dict_iterator)  # skip an int tensor
        dtype = next(state_dict_iterator).dtype
        model.to(dtype).load_state_dict(model_state_dict)
        model.additive_models[0].sync_tensor_maps()
        model.scaler.sync_tensor_maps()

        # Loading the metadata from the checkpoint
        model.metadata = merge_metadata(model.metadata, checkpoint.get("metadata"))

        return model

    def export(self) -> AtomisticModel:
        dtype = next(self.parameters()).dtype
        if dtype not in self.__supported_dtypes__:
            raise ValueError(f"unsupported dtype {self.dtype} for PhACE")

        # Make sure the model is all in the same dtype
        # For example, after training, the additive models could still be in
        # float64
        self.to(dtype)

        interaction_ranges = [
            self.hypers["cutoff"] * self.hypers["num_message_passing_layers"]
        ]
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

        return AtomisticModel(self.eval(), ModelMetadata(), capabilities)

    def _add_output(self, target_name: str, target_info: TargetInfo) -> None:
        self.outputs[target_name] = ModelOutput(
            quantity=target_info.quantity,
            unit=target_info.unit,
            per_atom=True,
            explicit_gradients=(["positions", "strain"] if target_info.quantity == "energy" and target_info.is_scalar and target_info.per_atom == False else [])
        )

        if target_name not in self.head_types:
            if target_info.is_scalar:
                use_mlp = True  # default to MLP for scalars
            else:
                use_mlp = False  # can't use MLP for equivariants
                # TODO: the equivariant could be a scalar...
        else:
            # specified by the user
            use_mlp = self.head_types[target_name] == "mlp"

        if use_mlp:
            if target_info.is_spherical or target_info.is_cartesian:
                raise ValueError("MLP heads are only supported for scalar targets.")
            
            layers = (
                [Linear(self.k_max_l[0], self.k_max_l[0]), torch.nn.SiLU()]
                if self.head_num_layers == 1
                else [Linear(self.k_max_l[0], 4 * self.k_max_l[0]), torch.nn.SiLU()]
                + [Linear(4 * self.k_max_l[0], 4 * self.k_max_l[0]), torch.nn.SiLU()] * (self.head_num_layers - 2)
                + [Linear(4 * self.k_max_l[0], self.k_max_l[0]), torch.nn.SiLU()]
            )
            self.heads[target_name] = torch.nn.Sequential(*layers)
        else:
            self.heads[target_name] = torch.nn.Identity()

        if target_info.is_scalar:
            # self.last_layers[target_name] = EquivariantLastLayer(
            #     [(0, 1)], , [[]], [target_info.layout.block(0).properties]
            # )
            # if [(0, 1)] not in self.requested_LS_tuples:
            #     self.requested_LS_tuples.append((0, 1))
            self.last_layers[target_name] = Linear(
                self.k_max_l[0],
                1
            )
        elif target_info.is_cartesian:
            # # here, we handle Cartesian targets
            # if len(target_info.layout.block().components) == 1:
            #     self.last_layers[target_name] = EquivariantLastLayer(
            #         [(1, 1)],
            #         self.k_max_l,
            #         [block.components for block in target_info.layout.blocks()],
            #         [block.properties for block in target_info.layout.blocks()],
            #     )
            #     self.requested_LS_tuples.append((1, 1))
            # else:
            #     raise NotImplementedError(
            #         "PhACE only supports Cartesian targets with rank=1."
            #     )
            pass
        else:  # spherical equivariant
            irreps = []
            for key in target_info.layout.keys:
                key_values = key.values
                L = int(key_values[0])
                S = int(key_values[1])
                irreps.append((L, S))
            self.last_layers[target_name] = EquivariantLastLayer(
                irreps,
                self.k_max_l,
                [block.components for block in target_info.layout.blocks()],
                [block.properties for block in target_info.layout.blocks()],
            )
            for irrep in irreps:
                if irrep not in self.requested_LS_tuples:
                    self.requested_LS_tuples.append(irrep)

        self.key_labels[target_name] = target_info.layout.keys
        self.component_labels[target_name] = [
            block.components for block in target_info.layout.blocks()
        ]
        self.property_labels[target_name] = [
            block.properties for block in target_info.layout.blocks()
        ]

    def requested_neighbor_lists(
        self,
    ) -> List[NeighborListOptions]:
        return [
            NeighborListOptions(
                cutoff=self.cutoff_radius,
                full_list=True,
                strict=True,
            )
        ]

    def get_checkpoint(self) -> Dict:
        checkpoint = {
            "architecture_name": "experimental.phace",
            "model_ckpt_version": self.__checkpoint_version__,
            "metadata": self.metadata,
            "model_data": {
                "model_hypers": self.hypers,
                "dataset_info": self.dataset_info,
            },
            "epoch": None,
            "best_epoch": None,
            "model_state_dict": self.state_dict(),
            "best_model_state_dict": self.state_dict(),
        }
        return checkpoint

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict) -> Dict:
        return ValueError()
