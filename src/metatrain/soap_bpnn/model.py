from pathlib import Path
from typing import Dict, List, Optional, Union

import featomic.torch
import metatensor.torch
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    System,
)
from metatensor.torch.learn.nn import Linear as LinearMap
from metatensor.torch.learn.nn import ModuleMap

from metatrain.utils.data import TargetInfo
from metatrain.utils.data.dataset import DatasetInfo

from ..utils.additive import ZBL, CompositionModel
from ..utils.dtype import dtype_to_str
from ..utils.long_range import DummyLongRangeFeaturizer, LongRangeFeaturizer
from ..utils.metadata import append_metadata_references
from ..utils.scaler import Scaler
from ..utils.sum_over_atoms import sum_over_atoms
from .spherical import TensorBasis


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: TensorMap) -> TensorMap:
        return x


class MLPMap(ModuleMap):
    def __init__(self, atomic_types: List[int], hypers: dict) -> None:
        # hardcoded for now, but could be a hyperparameter
        activation_function = torch.nn.SiLU()

        # Build a neural network for each species
        nns_per_species = []
        for _ in atomic_types:
            module_list: List[torch.nn.Module] = []
            for _ in range(hypers["num_hidden_layers"]):
                if len(module_list) == 0:
                    module_list.append(
                        torch.nn.Linear(
                            hypers["input_size"],
                            hypers["num_neurons_per_layer"],
                            bias=False,
                        )
                    )
                else:
                    module_list.append(
                        torch.nn.Linear(
                            hypers["num_neurons_per_layer"],
                            hypers["num_neurons_per_layer"],
                            bias=False,
                        )
                    )
                module_list.append(activation_function)

            nns_per_species.append(torch.nn.Sequential(*module_list))
        in_keys = Labels(
            "center_type",
            values=torch.tensor(atomic_types).reshape(-1, 1),
        )
        out_properties = [
            Labels(
                names=["properties"],
                values=torch.arange(
                    hypers["num_neurons_per_layer"],
                ).reshape(-1, 1),
            )
            for _ in range(len(in_keys))
        ]
        super().__init__(in_keys, nns_per_species, out_properties)
        self.activation_function = activation_function


class LayerNormMap(ModuleMap):
    def __init__(self, atomic_types: List[int], n_layer: int) -> None:
        # one layernorm for each species
        layernorm_per_species = []
        for _ in atomic_types:
            layernorm_per_species.append(torch.nn.LayerNorm((n_layer,)))

        in_keys = Labels(
            "center_type",
            values=torch.tensor(atomic_types).reshape(-1, 1),
        )
        out_properties = [
            Labels(
                names=["properties"],
                values=torch.arange(n_layer).reshape(-1, 1),
            )
            for _ in range(len(in_keys))
        ]
        super().__init__(in_keys, layernorm_per_species, out_properties)


class MLPHeadMap(ModuleMap):
    def __init__(
        self, in_keys: Labels, num_features: int, out_properties: List[Labels]
    ) -> None:
        # hardcoded for now, but could be a hyperparameter
        activation_function = torch.nn.SiLU()

        # Build a neural network for each species. 1 layer for now.
        nns_per_species = []
        for _ in in_keys:
            nns_per_species.append(
                torch.nn.Sequential(
                    torch.nn.Linear(num_features, num_features, bias=False),
                    activation_function,
                )
            )

        super().__init__(in_keys, nns_per_species, out_properties)
        self.activation_function = activation_function


class SoapBpnn(torch.nn.Module):
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float64, torch.float32]
    __default_metadata__ = ModelMetadata(
        references={
            "implementation": [
                "rascaline: https://github.com/Luthaf/rascaline",
            ],
            "architecture": [
                "SOAP: https://doi.org/10.1002/qua.24927",
                "BPNN: https://link.aps.org/doi/10.1103/PhysRevLett.98.146401",
            ],
        }
    )

    component_labels: Dict[str, List[List[Labels]]]  # torchscript needs this

    def __init__(self, model_hypers: Dict, dataset_info: DatasetInfo) -> None:
        super().__init__()
        self.hypers = model_hypers
        self.dataset_info = dataset_info
        self.atomic_types = dataset_info.atomic_types

        self.soap_calculator = featomic.torch.SoapPowerSpectrum(**self.hypers["soap"])

        soap_size = (
            (len(self.atomic_types) * (len(self.atomic_types) + 1) // 2)
            * (self.hypers["soap"]["basis"]["radial"]["max_radial"] + 1) ** 2
            * (self.hypers["soap"]["basis"]["max_angular"] + 1)
        )

        hypers_bpnn = {**self.hypers["bpnn"]}
        hypers_bpnn["input_size"] = soap_size

        if hypers_bpnn["layernorm"]:
            self.layernorm = LayerNormMap(self.atomic_types, soap_size)
        else:
            self.layernorm = Identity()

        self.bpnn = MLPMap(self.atomic_types, hypers_bpnn)

        self.neighbors_species_labels = Labels(
            names=["neighbor_1_type", "neighbor_2_type"],
            values=torch.combinations(
                torch.tensor(self.atomic_types, dtype=torch.int),
                with_replacement=True,
            ),
        )
        self.center_type_labels = Labels(
            names=["center_type"],
            values=torch.tensor(self.atomic_types).reshape(-1, 1),
        )

        if hypers_bpnn["num_hidden_layers"] == 0:
            self.n_inputs_last_layer = hypers_bpnn["input_size"]
        else:
            self.n_inputs_last_layer = hypers_bpnn["num_neurons_per_layer"]

        # long-range module
        self.nl_options = self.soap_calculator.requested_neighbor_lists()[0]
        if self.hypers["long_range"]["enable"]:
            self.long_range = True
            self.long_range_featurizer = LongRangeFeaturizer(
                self.hypers["long_range"],
                self.n_inputs_last_layer,
                self.nl_options,
            )
        else:
            self.long_range = False
            self.long_range_featurizer = DummyLongRangeFeaturizer()  # for torchscript

        self.last_layer_feature_size = self.n_inputs_last_layer * len(self.atomic_types)

        self.outputs = {
            "features": ModelOutput(unit="", per_atom=True)
        }  # the model is always capable of outputting the internal features

        self.single_label = Labels.single()

        self.num_properties: Dict[str, Dict[str, int]] = {}  # by target and block
        self.basis_calculators = torch.nn.ModuleDict({})
        self.heads = torch.nn.ModuleDict({})
        self.head_types = self.hypers["heads"]
        self.last_layers = torch.nn.ModuleDict({})
        self.key_labels: Dict[str, Labels] = {}
        self.component_labels: Dict[str, List[List[Labels]]] = {}
        self.property_labels: Dict[str, List[Labels]] = {}
        for target_name, target in dataset_info.targets.items():
            self._add_output(target_name, target)

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

    def restart(self, dataset_info: DatasetInfo) -> "SoapBpnn":
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
                "The SOAP-BPNN model does not support adding new atomic types."
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
        device = systems[0].positions.device
        if self.neighbors_species_labels.device != device:
            self.neighbors_species_labels = self.neighbors_species_labels.to(device)
        if self.center_type_labels.device != device:
            self.center_type_labels = self.center_type_labels.to(device)
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

        # initialize the return dictionary
        return_dict: Dict[str, TensorMap] = {}

        soap_features = self.soap_calculator(systems, selected_samples=selected_atoms)

        device = soap_features.block(0).values.device
        soap_features = soap_features.keys_to_properties(self.neighbors_species_labels)

        soap_features = self.layernorm(soap_features)
        features = self.bpnn(soap_features)

        if self.long_range:
            # slightly painful because:
            # - the features are split per center type
            # - we have to recompute the edge vectors again outside of featomic

            # first, send center_type to the samples dimension and make sure the
            # ordering is the same as in the systems
            merged_features = (
                metatensor.torch.sort(
                    features.keys_to_samples("center_type"), axes="samples"
                )
                .block()
                .values
            )

            # calculate the distances between all atoms
            # TODO: replace by something smarter once `register_autograd_neighbors`
            # is fixed
            edge_vectors = torch.concatenate(
                [
                    system.positions[
                        system.get_neighbor_list(self.nl_options).samples.values[:, 1]
                    ]
                    - system.positions[
                        system.get_neighbor_list(self.nl_options).samples.values[:, 0]
                    ]
                    + system.get_neighbor_list(self.nl_options)
                    .samples.values[:, 2:]
                    .to(system.cell.dtype)
                    @ system.cell
                    for system in systems
                ]
            )
            distances = torch.sqrt(torch.sum(edge_vectors**2, dim=-1))

            long_range_features_tensor = self.long_range_featurizer(
                systems, merged_features, distances
            )

            # also sort the original features to avoid problems
            features = metatensor.torch.sort(features, axes="samples")

            # split the long-range features back to center types
            center_types = torch.concatenate([system.types for system in systems])
            long_range_features = TensorMap(
                keys=features.keys,
                blocks=[
                    TensorBlock(
                        values=long_range_features_tensor[center_types == center_type],
                        samples=features.block(
                            {"center_type": int(center_type)}
                        ).samples,
                        components=features.block(
                            {"center_type": int(center_type)}
                        ).components,
                        properties=features.block(
                            {"center_type": int(center_type)}
                        ).properties,
                    )
                    for center_type in features.keys.column("center_type")
                ],
            )

            # combine short- and long-range features
            features = metatensor.torch.add(features, long_range_features)

        # output the hidden features, if requested:
        if "features" in outputs:
            features_options = outputs["features"]
            out_features = features.keys_to_properties(self.center_type_labels)
            if not features_options.per_atom:
                out_features = sum_over_atoms(out_features)
            return_dict["features"] = _remove_center_type_from_properties(out_features)

        features_by_output: Dict[str, TensorMap] = {}
        for output_name, head in self.heads.items():
            features_by_output[output_name] = head(features)

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
                f"mtt::{base_name}" not in features_by_output
                and base_name not in features_by_output
            ):
                raise ValueError(
                    f"Features {output_name} can only be requested "
                    f"if the corresponding output {base_name} is also requested."
                )
            if f"mtt::{base_name}" in features_by_output:
                base_name = f"mtt::{base_name}"
            features_options = outputs[output_name]
            out_features = features_by_output[base_name].keys_to_properties(
                self.center_type_labels
            )
            if not features_options.per_atom:
                out_features = sum_over_atoms(out_features)
            return_dict[output_name] = _remove_center_type_from_properties(out_features)

        atomic_properties: Dict[str, TensorMap] = {}
        for output_name, output_layers in self.last_layers.items():
            if output_name in outputs:
                blocks: List[TensorBlock] = []
                for layer_idx, (layer_key, output_layer) in enumerate(
                    output_layers.items()
                ):
                    components = self.component_labels[output_name][layer_idx]
                    properties = self.property_labels[output_name][layer_idx]
                    invariant_coefficients = output_layer(
                        features_by_output[output_name]
                    )
                    invariant_coefficients = invariant_coefficients.keys_to_samples(
                        "center_type"
                    )
                    tensor_basis = torch.tensor(0)
                    for (
                        output_name_basis,
                        basis_calculators_by_block,
                    ) in self.basis_calculators.items():
                        # need to loop again and do this due to torchscript
                        if output_name_basis == output_name:
                            for (
                                basis_calculator_key,
                                basis_calculator,
                            ) in basis_calculators_by_block.items():
                                if basis_calculator_key == layer_key:
                                    tensor_basis = basis_calculator(
                                        systems, selected_atoms
                                    )
                    # multiply the invariant coefficients by the elements of the
                    # tensor basis
                    invariant_coefficients_tensor = (
                        invariant_coefficients.block().values.reshape(
                            (
                                invariant_coefficients.block().values.shape[0],
                                len(properties),
                                tensor_basis.shape[2],
                            )
                        )
                    )
                    # [sample, property, basis], [sample, component, property] to
                    # [sample. component, property]
                    atomic_property_tensor = torch.einsum(
                        "spb, scb -> scp",
                        invariant_coefficients_tensor,
                        tensor_basis,
                    )
                    if len(components) == 0:
                        # "scalar", i.e. no components
                        atomic_property_tensor = atomic_property_tensor.squeeze(1)
                    blocks.append(
                        TensorBlock(
                            atomic_property_tensor,
                            invariant_coefficients.block().samples.remove(
                                "center_type"
                            ),
                            components,
                            properties,
                        )
                    )
                atomic_properties[output_name] = TensorMap(
                    self.key_labels[output_name], blocks
                )

        for output_name, atomic_property in atomic_properties.items():
            if outputs[output_name].per_atom:
                return_dict[output_name] = atomic_property
            else:
                # sum the atomic property to get the total property
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

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path]) -> "SoapBpnn":
        # Load the checkpoint
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        model_data = checkpoint["model_data"]
        model_state_dict = checkpoint["model_state_dict"]

        # Create the model
        model = cls(**model_data)
        dtype = next(iter(model_state_dict.values())).dtype
        model.to(dtype).load_state_dict(model_state_dict)
        model.additive_models[0].sync_tensor_maps()

        return model

    def export(
        self, metadata: Optional[ModelMetadata] = None
    ) -> MetatensorAtomisticModel:
        dtype = next(self.parameters()).dtype
        if dtype not in self.__supported_dtypes__:
            raise ValueError(f"unsupported dtype {self.dtype} for SoapBpnn")

        # Make sure the model is all in the same dtype
        # For example, after training, the additive models could still be in
        # float64
        self.to(dtype)

        # Additionally, the composition model contains some `TensorMap`s that cannot
        # be registered correctly with Pytorch. This funciton moves them:
        self.additive_models[0]._move_weights_to_device_and_dtype(
            torch.device("cpu"), torch.float64
        )

        interaction_ranges = [self.hypers["soap"]["cutoff"]["radius"]]
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

    def _add_output(self, target_name: str, target: TargetInfo) -> None:
        # register bases of spherical tensors (TensorBasis)
        self.num_properties[target_name] = {}
        self.basis_calculators[target_name] = torch.nn.ModuleDict({})
        if target.is_scalar:
            for key, block in target.layout.items():
                dict_key = target_name
                for n, k in zip(key.names, key.values):
                    dict_key += f"_{n}_{int(k)}"
                self.num_properties[target_name][dict_key] = len(
                    block.properties.values
                )
                self.basis_calculators[target_name][dict_key] = TensorBasis(
                    self.atomic_types, self.hypers["soap"], o3_lambda=0, o3_sigma=1
                )
        elif target.is_spherical:
            for key, block in target.layout.items():
                dict_key = target_name
                for n, k in zip(key.names, key.values):
                    dict_key += f"_{n}_{int(k)}"
                self.num_properties[target_name][dict_key] = len(
                    block.properties.values
                )
                o3_lambda = int(key[0])
                o3_sigma = int(key[1])
                self.basis_calculators[target_name][dict_key] = TensorBasis(
                    self.atomic_types, self.hypers["soap"], o3_lambda, o3_sigma
                )
        else:
            raise ValueError("SOAP-BPNN only supports scalar and spherical targets.")

        if target_name not in self.head_types:  # default to linear head
            self.heads[target_name] = Identity()
        elif self.head_types[target_name] == "mlp":
            if not target.is_scalar:
                raise ValueError(
                    "MLP head is only supported for scalar targets, "
                    f"but target {target_name} is not scalar."
                )
            self.heads[target_name] = MLPHeadMap(
                in_keys=Labels(
                    "center_type",
                    values=torch.tensor(self.atomic_types).reshape(-1, 1),
                ),
                num_features=self.n_inputs_last_layer,
                out_properties=[
                    Labels(
                        names=["property"],
                        values=torch.arange(self.n_inputs_last_layer).reshape(-1, 1),
                    )
                    for _ in self.atomic_types
                ],
            )
        elif self.head_types[target_name] == "linear":
            self.heads[target_name] = Identity()
        else:
            raise ValueError(
                f"Unsupported head type {self.head_types[target_name]} "
                f"for target {target_name}"
            )

        ll_features_name = (
            f"mtt::aux::{target_name.replace('mtt::', '')}_last_layer_features"
        )
        self.outputs[ll_features_name] = ModelOutput(per_atom=True)

        # last linear layers, one per block
        self.last_layers[target_name] = torch.nn.ModuleDict({})
        for key, block in target.layout.items():
            dict_key = target_name
            for n, k in zip(key.names, key.values):
                dict_key += f"_{n}_{int(k)}"
            # the spherical tensor basis is made of 2*l+1 tensors, same as the number
            # of components
            out_properties = Labels.range(
                "property",
                len(block.properties.values)
                * (1 if target.is_scalar else len(block.components[0])),
            )
            last_layer_arguments = {
                "in_keys": Labels(
                    "center_type",
                    values=torch.tensor(self.atomic_types).reshape(-1, 1),
                ),
                "in_features": self.n_inputs_last_layer,
                "out_features": len(block.properties.values)
                * (1 if target.is_scalar else len(block.components[0])),
                "bias": False,
                "out_properties": [out_properties for _ in self.atomic_types],
            }
            self.last_layers[target_name][dict_key] = LinearMap(**last_layer_arguments)

        self.key_labels[target_name] = target.layout.keys
        self.component_labels[target_name] = [
            block.components for block in target.layout.blocks()
        ]
        self.property_labels[target_name] = [
            block.properties for block in target.layout.blocks()
        ]

        self.outputs[target_name] = ModelOutput(
            quantity=target.quantity,
            unit=target.unit,
            per_atom=True,
        )


def _remove_center_type_from_properties(tensor_map: TensorMap) -> TensorMap:
    new_blocks: List[TensorBlock] = []
    for block in tensor_map.blocks():
        new_blocks.append(
            TensorBlock(
                values=block.values,
                samples=block.samples,
                components=block.components,
                properties=Labels(
                    names=["properties"],
                    values=torch.arange(
                        block.values.shape[-1], device=block.values.device
                    ).reshape(-1, 1),
                ),
            )
        )
    return TensorMap(keys=tensor_map.keys, blocks=new_blocks)
