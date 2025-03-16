import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import metatensor.torch
import numpy as np
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

from .modules.layers import Linear

from ...utils.additive import ZBL, CompositionModel
from ...utils.data.dataset import DatasetInfo, TargetInfo
from ...utils.dtype import dtype_to_str
from ...utils.io import check_file_extension
from ...utils.scaler import Scaler
from .modules.center_embedding import embed_centers, embed_centers_tensor_map
from .modules.cg import get_cg_coefficients
from .modules.cg_iterator import CGIterator
from .modules.initial_features import get_initial_features
from .modules.layers import EquivariantLastLayer, Identity, InvariantMLP
from .modules.message_passing import EquivariantMessagePasser, InvariantMessagePasser
from .modules.precomputations import Precomputer
from .modules.tensor_product import (
    couple_features,
    uncouple_features,
)
from .utils import systems_to_batch


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=(
        "The TorchScript type system doesn't support " "instance-level annotations"
    ),
)


class PhACE(torch.nn.Module):

    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float64, torch.float32]

    component_labels: Dict[str, List[List[Labels]]]
    U_dict: Dict[int, torch.Tensor]
    U_dict_parity: Dict[str, torch.Tensor]

    def __init__(self, model_hypers: Dict, dataset_info: DatasetInfo) -> None:
        super().__init__()
        self.hypers = model_hypers
        self.dataset_info = dataset_info
        self.new_outputs = list(dataset_info.targets.keys())
        self.atomic_types = sorted(dataset_info.atomic_types)

        self.cutoff_radius = model_hypers["cutoff"]
        self.dataset_info = dataset_info
        self.model_hypers = model_hypers

        self.nu_scaling = model_hypers["nu_scaling"]
        self.mp_scaling = model_hypers["mp_scaling"]
        self.overall_scaling = model_hypers["overall_scaling"]

        n_channels = model_hypers["num_element_channels"]

        # Embedding of the atomic types
        self.embeddings = torch.nn.Embedding(len(self.atomic_types), n_channels)

        # A buffer that maps atomic types to indices in the embeddings
        species_to_species_index = torch.zeros(
            (max(self.atomic_types) + 1,), dtype=torch.int
        )
        species_to_species_index[self.atomic_types] = torch.arange(
            len(self.atomic_types), dtype=torch.int
        )
        self.register_buffer("species_to_species_index", species_to_species_index)

        self.nu_max = model_hypers["max_correlation_order_per_layer"]
        self.num_message_passing_layers = model_hypers["num_message_passing_layers"]
        if self.num_message_passing_layers < 1:
            raise ValueError("Number of message-passing layers must be at least 1")

        # The message passing is invariant for the first layer
        self.invariant_message_passer = InvariantMessagePasser(
            model_hypers,
            self.atomic_types,
            self.mp_scaling,
            model_hypers["disable_nu_0"],
        )

        self.atomic_types = self.atomic_types
        n_max = self.invariant_message_passer.n_max_l
        self.l_max = len(n_max) - 1
        self.k_max_l = [
            n_channels * n_max[l] for l in range(self.l_max + 1)  # noqa: E741
        ]
        print(self.k_max_l)
        self.k_max_l_max = [0] * (self.l_max + 1)
        previous = 0
        for l in range(self.l_max, -1, -1):
            self.k_max_l_max[l] = self.k_max_l[l] - previous
            previous = self.k_max_l[l]
        print(self.k_max_l_max)

        cgs = get_cg_coefficients(self.l_max)
        cgs = {
            str(l1) + "_" + str(l2) + "_" + str(L): tensor
            for (l1, l2, L), tensor in cgs._cgs.items()
        }

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

        # A module that precomputes quantities that are useful in all message-passing
        # steps (spherical harmonics, distances)
        self.precomputer = Precomputer(
            self.l_max, use_sphericart=model_hypers["use_sphericart"]
        )

        self.cg_iterator = CGIterator(
            self.k_max_l_max,
            self.nu_max - 1,
        )

        cg_calculator = get_cg_coefficients(2 * ((self.l_max + 1) // 2))
        print(self.l_max)
        self.padded_l_list = [2 * ((l + 1) // 2) for l in range(self.l_max + 1)]
        self.U_dict = {}
        for padded_l in np.unique(self.padded_l_list):
            cg_tensors = [
                cg_calculator._cgs[(padded_l // 2, padded_l // 2, L)]
                for L in range(padded_l + 1)
            ]
            U = torch.concatenate(
                [cg_tensor for cg_tensor in cg_tensors], dim=2
            ).reshape((padded_l + 1) ** 2, (padded_l + 1) ** 2)
            assert torch.allclose(
                U @ U.T, torch.eye((padded_l + 1) ** 2, dtype=U.dtype)
            )
            assert torch.allclose(
                U.T @ U, torch.eye((padded_l + 1) ** 2, dtype=U.dtype)
            )
            self.U_dict[padded_l] = U

        self.U_dict_parity: Dict[str, torch.Tensor] = {}
        for padded_l in list(self.U_dict.keys()):
            self.U_dict_parity[f"{padded_l}_{1}"] = self.U_dict[padded_l].clone()
            # mask out odd l values
            for l in range(1, padded_l + 1, 2):
                # print(l, padded_l)
                self.U_dict_parity[f"{padded_l}_{1}"][:, l**2:(l + 1)**2] = 0.0
                # print(self.U_dict_parity[f"{padded_l}_{1}"])
            self.U_dict_parity[f"{padded_l}_{-1}"] = self.U_dict[padded_l].clone()
            # mask out even l values
            for l in range(0, padded_l + 1, 2):
                # print(l, padded_l)
                self.U_dict_parity[f"{padded_l}_{-1}"][:, l**2:(l + 1)**2] = 0.0

        # Subsequent message-passing layers
        equivariant_message_passers: List[EquivariantMessagePasser] = []
        generalized_cg_iterators: List[CGIterator] = []
        for _ in range(self.num_message_passing_layers - 1):
            equivariant_message_passer = EquivariantMessagePasser(
                model_hypers,
                self.atomic_types,
                self.padded_l_list,
                self.mp_scaling,
            )
            equivariant_message_passers.append(equivariant_message_passer)
            generalized_cg_iterator = CGIterator(
                self.k_max_l_max,
                self.nu_max - 1,
            )
            generalized_cg_iterators.append(generalized_cg_iterator)
        self.equivariant_message_passers = torch.nn.ModuleList(
            equivariant_message_passers
        )
        self.generalized_cg_iterators = torch.nn.ModuleList(generalized_cg_iterators)

        self.last_layer_feature_size = self.k_max_l[0]

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

        self.single_label = Labels.single()

        self.intermediate_linears = torch.nn.ModuleList(
            [
                Linear(self.k_max_l[l], self.k_max_l[l])
                for l in range(self.l_max + 1)
            ]
        )

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
        if self.U_dict[0].device != device:
            self.U_dict = {
                padded_l: U.to(device) for padded_l, U in self.U_dict.items()
            }
            self.U_dict_parity = {
                key: U.to(device) for key, U in self.U_dict_parity.items()
            }

        dtype = systems[0].dtype
        if self.U_dict[0].dtype != dtype:
            self.U_dict = {padded_l: U.to(dtype) for padded_l, U in self.U_dict.items()}
            self.U_dict_parity = {
                key: U.to(dtype) for key, U in self.U_dict_parity.items()
            }

        neighbor_list_options = self.requested_neighbor_lists()[0]  # there is only one
        structures = systems_to_batch(systems, neighbor_list_options)

        n_atoms = len(structures["positions"])

        # precomputation of distances and spherical harmonics
        r, sh = self.precomputer(
            positions=structures["positions"],
            cells=structures["cells"],
            species=structures["species"],
            cell_shifts=structures["cell_shifts"],
            pairs=structures["pairs"],
            structure_pairs=structures["structure_pairs"],
            structure_offsets=structures["structure_offsets"],
        )

        # scaling the spherical harmonics in this way makes sure that each successive
        # body-order is scaled by the same factor
        sh = metatensor.torch.multiply(sh, self.nu_scaling)

        # compute sample labels
        samples_values = torch.stack(
            (
                structures["structure_centers"],
                structures["centers"],
                structures["species"],
            ),
            dim=1,
        )
        samples = metatensor.torch.Labels(
            names=["system", "atom", "center_type"],
            values=samples_values,
        )

        # calculate the center embeddings; these are shared across all layers
        center_species_indices = self.species_to_species_index[structures["species"]]
        center_embeddings = self.embeddings(center_species_indices)

        initial_features = get_initial_features(
            structures["structure_centers"],
            structures["centers"],
            structures["species"],
            structures["positions"].dtype,
            self.k_max_l[0],
        )  # these features are all one
        initial_element_embedding = embed_centers_tensor_map(
            initial_features, center_embeddings
        )
        # now they are all the same as the center embeddings

        # ACE-like features
        spherical_expansion = self.invariant_message_passer(
            r,
            sh,
            structures["structure_offsets"][structures["structure_pairs"]]
            + structures["pairs"][:, 0],
            structures["structure_offsets"][structures["structure_pairs"]]
            + structures["pairs"][:, 1],
            n_atoms,
            initial_element_embedding,
            samples,
        )

        split_features: List[List[torch.Tensor]] = []
        for l in range(self.l_max, -1, -1):
            lower_bound = self.k_max_l[l + 1] if l < self.l_max else 0
            upper_bound = self.k_max_l[l]
            split_features = [
                [
                    spherical_expansion.block({"o3_lambda": lp}).values[
                        :, :, lower_bound:upper_bound
                    ]
                    for lp in range(l + 1)
                ]
            ] + split_features

        uncoupled_features: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for l in range(self.l_max + 1):
            uncoupled_features.append(
                uncouple_features(
                    split_features[l],
                    (self.U_dict_parity[f"{self.padded_l_list[l]}_{1}"], self.U_dict_parity[f"{self.padded_l_list[l]}_{-1}"]),
                    self.padded_l_list[l],
                )
            )

        features = self.cg_iterator(uncoupled_features)

        coupled_features_0: List[List[torch.Tensor]] = []
        for l in range(self.l_max + 1):
            coupled_features_0.append(
                couple_features(
                    features[l],
                    (self.U_dict_parity[f"{self.padded_l_list[l]}_{1}"], self.U_dict_parity[f"{self.padded_l_list[l]}_{-1}"]),
                    self.padded_l_list[l],
                )[0]
            )

        concatenated_coupled_features_0 = []
        for l in range(self.l_max + 1):
            concatenated_coupled_features_0.append(
                torch.concatenate(
                    [coupled_features_0[lp][l] for lp in range(l, self.l_max + 1)], dim=-1
                )
            )

        for l, linear in enumerate(self.intermediate_linears):
            concatenated_coupled_features_0[l] = linear(concatenated_coupled_features_0[l])

        coupled_features_0: List[List[torch.Tensor]] = []
        for l in range(self.l_max, -1, -1):
            lower_bound = self.k_max_l[l + 1] if l < self.l_max else 0
            upper_bound = self.k_max_l[l]
            coupled_features_0 = [
                [
                    concatenated_coupled_features_0[lp][
                        :, :, lower_bound:upper_bound
                    ]
                    for lp in range(l + 1)
                ]
            ] + coupled_features_0

        uncoupled_features_0: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for l in range(self.l_max + 1):
            uncoupled_features_0.append(
                uncouple_features(
                    coupled_features_0[l],
                    (self.U_dict_parity[f"{self.padded_l_list[l]}_{1}"], self.U_dict_parity[f"{self.padded_l_list[l]}_{-1}"]),
                    self.padded_l_list[l],
                )
            )
        features = uncoupled_features_0

        # message passing
        for message_passer, generalized_cg_iterator in zip(
            self.equivariant_message_passers, self.generalized_cg_iterators
        ):
            embedded_features = embed_centers(features, center_embeddings)
            mp_features = message_passer(
                r,
                sh,
                structures["structure_offsets"][structures["structure_pairs"]]
                + structures["pairs"][:, 0],
                structures["structure_offsets"][structures["structure_pairs"]]
                + structures["pairs"][:, 1],
                embedded_features,
                self.U_dict_parity,
            )
            iterated_features = generalized_cg_iterator(mp_features)
            features = iterated_features

        # TODO: change position?
        features = embed_centers(features, center_embeddings)

        coupled_features: List[List[torch.Tensor]] = []
        for l in range(self.l_max + 1):
            coupled_features.append(
                couple_features(
                    features[l],
                    (self.U_dict_parity[f"{self.padded_l_list[l]}_{1}"], self.U_dict_parity[f"{self.padded_l_list[l]}_{-1}"]),
                    self.padded_l_list[l],
                )[0]
            )

        concatenated_coupled_features = []
        for l in range(self.l_max + 1):
            concatenated_coupled_features.append(
                torch.concatenate(
                    [coupled_features[lp][l] for lp in range(l, self.l_max + 1)], dim=-1
                )
            )

        features = TensorMap(
            keys=spherical_expansion.keys,
            blocks=[
                TensorBlock(
                    values=concatenated_coupled_features[l],
                    samples=spherical_expansion.block({"o3_lambda": l}).samples,
                    components=spherical_expansion.block({"o3_lambda": l}).components,
                    properties=spherical_expansion.block({"o3_lambda": l}).properties,
                )
                for l in range(self.l_max + 1)
            ],
        )

        # remove the center_type dimension
        features = metatensor.torch.remove_dimension(features, "samples", "center_type")

        if selected_atoms is not None:
            features = metatensor.torch.slice(
                features, axis="samples", selection=selected_atoms
            )

        return_dict: Dict[str, TensorMap] = {}

        # output the hidden features, if requested (invariant only):
        if "features" in outputs:
            feature_tmap = TensorMap(
                keys=self.single_label,
                blocks=[
                    TensorBlock(
                        values=features.block(
                            {"o3_lambda": 0, "o3_sigma": 1}
                        ).values.squeeze(1),
                        samples=features.block({"o3_lambda": 0, "o3_sigma": 1}).samples,
                        components=[],
                        properties=features.block(
                            {"o3_lambda": 0, "o3_sigma": 1}
                        ).properties,
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

        for output_name, output_head in self.heads.items():
            if output_name in outputs:
                return_dict[output_name] = output_head(features)

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
                    output_layer(return_dict[output_name]),
                    self.overall_scaling,
                )

        for output_name in self.last_layers:
            if output_name in outputs:
                if not outputs[output_name].per_atom:
                    return_dict[output_name] = metatensor.torch.sum_over_samples(
                        return_dict[output_name], ["atom"]
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
    def load_checkpoint(cls, path: Union[str, Path]) -> "PhACE":

        # Load the checkpoint
        checkpoint = torch.load(path)
        model_hypers = checkpoint["model_hypers"]
        model_state_dict = checkpoint["model_state_dict"]

        # Create the model
        model = cls(**model_hypers)
        dtype = next(iter(model_state_dict["embeddings.weight"])).dtype
        model.to(dtype).load_state_dict(model_state_dict)

        return model

    def export(self) -> MetatensorAtomisticModel:
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

        return MetatensorAtomisticModel(self.eval(), ModelMetadata(), capabilities)

    def _add_output(self, target_name: str, target_info: TargetInfo) -> None:

        if target_info.is_cartesian:
            raise NotImplementedError("PhACE does not support Cartesian targets.")

        if target_name not in self.head_types:
            if target_info.is_scalar:
                use_mlp = True  # default to MLP for scalars
            else:
                use_mlp = False  # can't use MLP for equivariants
                # TODO: the equivariant could be a scalar...
        else:
            # specified by the user
            use_mlp = self.head_types[target_name] == "mlp"

        self.outputs[target_name] = ModelOutput(
            quantity=target_info.quantity,
            unit=target_info.unit,
            per_atom=True,
        )

        if use_mlp:
            if target_info.is_spherical:
                raise ValueError("MLP heads are not supported for spherical targets.")
            self.heads[target_name] = InvariantMLP(
                self.k_max_l[0], self.head_num_layers
            )
        else:
            self.heads[target_name] = Identity()

        if target_info.is_scalar:
            self.last_layers[target_name] = EquivariantLastLayer(
                [(0, 1)], self.k_max_l, [[]], [target_info.layout.block(0).properties]
            )
            if [(0, 1)] not in self.requested_LS_tuples:
                self.requested_LS_tuples.append((0, 1))
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

    def save_checkpoint(self, path: Union[str, Path]):
        torch.save(
            {
                "model_model_hypers": {
                    "model_model_hypers": self.model_hypers,
                    "dataset_info": self.dataset_info,
                },
                "model_state_dict": self.state_dict(),
            },
            check_file_extension(path, ".ckpt"),
        )
