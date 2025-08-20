import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Literal

from metatrain.utils.metadata import merge_metadata
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
import logging

from .modules.tensor_product import TensorProduct

from metatrain.utils.abc import ModelInterface
from metatrain.utils.additive import ZBL, CompositionModel
from metatrain.utils.data.dataset import DatasetInfo, TargetInfo
from metatrain.utils.dtype import dtype_to_str
from metatrain.utils.io import check_file_extension
from metatrain.utils.scaler import Scaler
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
from metatomic.torch import AtomisticModel


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=(
        "The TorchScript type system doesn't support " "instance-level annotations"
    ),
)


class PhACE(ModelInterface):
    __checkpoint_version__ = 1
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float64, torch.float32]
    __default_metadata__ = ModelMetadata(
        references={}
    )

    component_labels: Dict[str, List[List[Labels]]]
    U_dict: Dict[int, torch.Tensor]
    U_dict_parity: Dict[str, torch.Tensor]

    def __init__(self, hypers: Dict, dataset_info: DatasetInfo) -> None:
        super().__init__(hypers, dataset_info, self.__default_metadata__)
        
        self.hypers = hypers
        self.dataset_info = dataset_info
        self.new_outputs = list(dataset_info.targets.keys())
        self.atomic_types = sorted(dataset_info.atomic_types)

        self.cutoff_radius = hypers["cutoff"]
        self.dataset_info = dataset_info
        self.hypers = hypers

        self.nu_scaling = hypers["nu_scaling"]
        self.mp_scaling = hypers["mp_scaling"]
        self.overall_scaling = hypers["overall_scaling"]

        n_channels = hypers["num_element_channels"]

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

        self.nu_max = hypers["max_correlation_order_per_layer"]
        self.num_message_passing_layers = hypers["num_message_passing_layers"]
        if self.num_message_passing_layers < 1:
            raise ValueError("Number of message-passing layers must be at least 1")

        # The message passing is invariant for the first layer
        self.invariant_message_passer = InvariantMessagePasser(
            hypers,
            self.atomic_types,
            self.mp_scaling,
            hypers["disable_nu_0"],
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
            self.l_max, use_sphericart=hypers["use_sphericart"]
        )

        tensor_product = TensorProduct(self.k_max_l)

        self.cg_iterator = CGIterator(
            tensor_product,
            self.nu_max - 1,
        )

        # Subsequent message-passing layers
        equivariant_message_passers: List[EquivariantMessagePasser] = []
        generalized_cg_iterators: List[CGIterator] = []
        for _ in range(self.num_message_passing_layers - 1):
            equivariant_message_passer = EquivariantMessagePasser(
                hypers,
                self.atomic_types,
                tensor_product,
                self.mp_scaling,
            )
            equivariant_message_passers.append(equivariant_message_passer)
            generalized_cg_iterator = CGIterator(
                tensor_product,
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

        features = [spherical_expansion.block({"o3_lambda": l}).values for l in range(self.l_max + 1)]
        features = self.cg_iterator(features)

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
            )
            iterated_features = generalized_cg_iterator(mp_features)
            features = iterated_features

        # TODO: change position?
        embedded_features = embed_centers(features, center_embeddings)

        features = TensorMap(
            keys=spherical_expansion.keys,
            blocks=[
                TensorBlock(
                    values=embedded_features[l],
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
                                properties=self.property_labels[output_name][0]
                            )
                        ]
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

    def requested_neighbor_lists(
        self,
    ) -> List[NeighborListOptions]:
        return [self.requested_nl]

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> "SoapBpnn":
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
        elif target_info.is_cartesian:
            # here, we handle Cartesian targets
            if len(target_info.layout.block().components) == 1:
                self.last_layers[target_name] = EquivariantLastLayer(
                    [(1, 1)],
                    self.k_max_l,
                    [block.components for block in target_info.layout.blocks()],
                    [block.properties for block in target_info.layout.blocks()],
                )
                self.requested_LS_tuples.append((1, 1))
            else:
                raise NotImplementedError("PhACE only supports Cartesian targets with rank=1.")
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
