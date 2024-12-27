import copy
from pathlib import Path
from typing import Dict, List, Optional, Union

import metatensor.torch
import rascaline.torch
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

from ...utils.additive import ZBL, CompositionModel
from ...utils.dtype import dtype_to_str
from ...utils.scaler import Scaler


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: TensorMap) -> TensorMap:
        return x


class IdentityWithExtraArg(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s: List[System], x: TensorMap) -> TensorMap:
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
                            hypers["input_size"], hypers["num_neurons_per_layer"]
                        )
                    )
                else:
                    module_list.append(
                        torch.nn.Linear(
                            hypers["num_neurons_per_layer"],
                            hypers["num_neurons_per_layer"],
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
                    torch.nn.Linear(num_features, num_features),
                    activation_function,
                )
            )

        super().__init__(in_keys, nns_per_species, out_properties)
        self.activation_function = activation_function


class VectorFeaturizer(torch.nn.Module):
    def __init__(self, atomic_types, num_features, soap_hypers) -> None:
        super().__init__()
        self.atomic_types = atomic_types
        soap_vector_hypers = copy.deepcopy(soap_hypers)
        soap_vector_hypers["max_angular"] = 1
        self.soap_calculator = rascaline.torch.SphericalExpansion(
            radial_basis={"Gto": {}}, **soap_vector_hypers
        )
        self.neighbors_species_labels = Labels(
            names=["neighbor_type"],
            values=torch.tensor(self.atomic_types).reshape(-1, 1),
        )
        self.linear_layer = LinearMap(
            Labels(
                names=["o3_lambda", "o3_sigma", "center_type"],
                values=torch.stack(
                    [
                        torch.tensor([1] * len(self.atomic_types)),
                        torch.tensor([1] * len(self.atomic_types)),
                        torch.tensor(self.atomic_types),
                    ],
                    dim=1,
                ),
            ),
            in_features=soap_vector_hypers["max_radial"] * len(self.atomic_types),
            out_features=num_features,
            bias=False,
            out_properties=[
                Labels(
                    names=["property"],
                    values=torch.arange(num_features).reshape(-1, 1),
                )
                for _ in self.atomic_types
            ],
        )

    def forward(self, systems: List[System], scalar_features: TensorMap) -> TensorMap:
        device = scalar_features.block(0).values.device

        spherical_expansion = self.soap_calculator(systems)
        spherical_expansion = spherical_expansion.keys_to_properties(
            self.neighbors_species_labels.to(device)
        )

        # drop all l=0 blocks
        keys_to_drop_list: List[List[int]] = []
        for key in spherical_expansion.keys.values:
            o3_lambda = int(key[0])
            o3_sigma = int(key[1])
            center_species = int(key[2])
            if o3_lambda == 0 and o3_sigma == 1:
                keys_to_drop_list.append([o3_lambda, o3_sigma, center_species])
        keys_to_drop = Labels(
            names=["o3_lambda", "o3_sigma", "center_type"],
            values=torch.tensor(keys_to_drop_list, device=device),
        )
        spherical_expansion = metatensor.torch.drop_blocks(
            spherical_expansion, keys=keys_to_drop
        )
        vector_features = self.linear_layer(spherical_expansion)

        overall_features = metatensor.torch.TensorMap(
            keys=vector_features.keys,
            blocks=[
                TensorBlock(
                    values=scalar_features.block(
                        {"center_type": int(ct)}
                    ).values.unsqueeze(1)
                    * vector_features.block({"center_type": int(ct)}).values
                    * 100.0,
                    samples=vector_features.block({"center_type": int(ct)}).samples,
                    components=vector_features.block(
                        {"center_type": int(ct)}
                    ).components,
                    properties=vector_features.block(
                        {"center_type": int(ct)}
                    ).properties,
                )
                for ct in vector_features.keys.column("center_type")
            ],
        )

        return overall_features


class SoapBpnn(torch.nn.Module):

    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float64, torch.float32]

    def __init__(self, model_hypers: Dict, dataset_info: DatasetInfo) -> None:
        super().__init__()
        self.hypers = model_hypers
        self.dataset_info = dataset_info
        self.atomic_types = dataset_info.atomic_types

        self.soap_calculator = rascaline.torch.SoapPowerSpectrum(
            radial_basis={"Gto": {}}, **self.hypers["soap"]
        )

        soap_size = (
            (len(self.atomic_types) * (len(self.atomic_types) + 1) // 2)
            * self.hypers["soap"]["max_radial"] ** 2
            * (self.hypers["soap"]["max_angular"] + 1)
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

        self.last_layer_feature_size = self.n_inputs_last_layer * len(self.atomic_types)

        self.outputs = {
            "features": ModelOutput(unit="", per_atom=True)
        }  # the model is always capable of outputting the internal features
        for target_name in dataset_info.targets.keys():
            # the model can always output the last-layer features for the targets
            ll_features_name = (
                f"mtt::aux::{target_name.replace('mtt::', '')}_last_layer_features"
            )
            self.outputs[ll_features_name] = ModelOutput(per_atom=True)

        self.vector_featurizers = torch.nn.ModuleDict({})
        self.heads = torch.nn.ModuleDict({})
        self.head_types = self.hypers["heads"]
        self.last_layers = torch.nn.ModuleDict({})
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
                    if CompositionModel.is_valid_target(target_info)
                },
            ),
        )
        additive_models = [composition_model]
        if self.hypers["zbl"]:
            additive_models.append(ZBL(model_hypers, dataset_info))
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
                    if CompositionModel.is_valid_target(target_info)
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
        # initialize the return dictionary
        return_dict: Dict[str, TensorMap] = {}

        soap_features = self.soap_calculator(systems, selected_samples=selected_atoms)

        device = soap_features.block(0).values.device
        soap_features = soap_features.keys_to_properties(
            self.neighbors_species_labels.to(device)
        )

        soap_features = self.layernorm(soap_features)
        features = self.bpnn(soap_features)

        # output the hidden features, if requested:
        if "features" in outputs:
            features_options = outputs["features"]
            out_features = features.keys_to_properties(
                self.center_type_labels.to(device)
            )
            if not features_options.per_atom:
                out_features = metatensor.torch.sum_over_samples(out_features, ["atom"])
            return_dict["features"] = _remove_center_type_from_properties(out_features)

        features_by_output: Dict[str, TensorMap] = {}
        for output_name, vector_featurizer in self.vector_featurizers.items():
            features_by_output[output_name] = vector_featurizer(systems, features)
        for output_name, head in self.heads.items():
            features_by_output[output_name] = head(features_by_output[output_name])

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
                self.center_type_labels.to(device)
            )
            if not features_options.per_atom:
                out_features = metatensor.torch.sum_over_samples(out_features, ["atom"])
            return_dict[output_name] = _remove_center_type_from_properties(out_features)

        atomic_properties: Dict[str, TensorMap] = {}
        for output_name, output_layer in self.last_layers.items():
            if output_name in outputs:
                atomic_properties[output_name] = output_layer(
                    features_by_output[output_name]
                )

        for output_name, atomic_property in atomic_properties.items():
            atomic_property = atomic_property.keys_to_samples("center_type")
            if outputs[output_name].per_atom:
                # this operation should just remove the center_type label
                return_dict[output_name] = metatensor.torch.remove_dimension(
                    atomic_property, axis="samples", name="center_type"
                )
            else:
                # sum the atomic property to get the total property
                return_dict[output_name] = metatensor.torch.sum_over_samples(
                    atomic_property, ["atom", "center_type"]
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
    def load_checkpoint(cls, path: Union[str, Path]) -> "SoapBpnn":

        # Load the checkpoint
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        model_hypers = checkpoint["model_hypers"]
        model_state_dict = checkpoint["model_state_dict"]

        # Create the model
        model = cls(**model_hypers)
        dtype = next(iter(model_state_dict.values())).dtype
        model.to(dtype).load_state_dict(model_state_dict)

        return model

    def export(self) -> MetatensorAtomisticModel:
        dtype = next(self.parameters()).dtype
        if dtype not in self.__supported_dtypes__:
            raise ValueError(f"unsupported dtype {self.dtype} for SoapBpnn")

        # Make sure the model is all in the same dtype
        # For example, after training, the additive models could still be in
        # float64
        self.to(dtype)

        interaction_ranges = [self.hypers["soap"]["cutoff"]]
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

        return MetatensorAtomisticModel(self.eval(), ModelMetadata(), capabilities)

    def _add_output(self, target_name: str, target: TargetInfo) -> None:

        # featurizers for non-scalars
        if target.is_scalar:
            self.vector_featurizers[target_name] = IdentityWithExtraArg()
        elif target.is_spherical:
            values_list: List[List[int]] = target.layout.keys.values.tolist()
            if values_list != [[1, 1]]:
                raise ValueError(
                    "SOAP-BPNN only supports spherical targets with "
                    "`o3_lambda=1` and `o3_sigma=1`, "
                )
            self.vector_featurizers[target_name] = VectorFeaturizer(
                atomic_types=self.atomic_types,
                num_features=self.n_inputs_last_layer,
                soap_hypers=self.hypers["soap"],
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

        # last linear layer
        last_layer_arguments = {
            "in_features": self.n_inputs_last_layer,
            "out_features": len(target.layout.block().properties.values),
            "bias": False,
            "out_properties": [
                target.layout.block().properties for _ in self.atomic_types
            ],
        }
        if target.is_scalar:
            last_layer_arguments["in_keys"] = Labels(
                "center_type",
                values=torch.tensor(self.atomic_types).reshape(-1, 1),
            )
        else:  # spherical vector
            last_layer_arguments["in_keys"] = Labels(
                names=["o3_lambda", "o3_sigma", "center_type"],
                values=torch.stack(
                    [
                        torch.tensor([1] * len(self.atomic_types)),
                        torch.tensor([1] * len(self.atomic_types)),
                        torch.tensor(self.atomic_types),
                    ],
                    dim=1,
                ),
            )
        self.last_layers[target_name] = LinearMap(**last_layer_arguments)

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
