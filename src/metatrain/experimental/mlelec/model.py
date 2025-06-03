from pathlib import Path
from typing import Dict, List, Optional, Union

import metatensor.torch
import torch
import numpy as np

from sklearn.linear_model import RidgeCV

from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.learn.nn import EquivariantLinear
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)

from metatrain.utils.abc import ModelInterface
from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.dtype import dtype_to_str
from metatrain.utils.metadata import merge_metadata

# from ...utils.scaler import Scaler
from metatrain.utils.sum_over_atoms import sum_over_atoms

# from .modules.structures import concatenate_structures

# from .modules.utils import split_node_edge_targets
from .modules.descriptors import get_descriptor_calculator


class MLElec(ModelInterface):

    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float64, torch.float32]
    __default_metadata__ = ModelMetadata(
        references={"architecture": ["https://arxiv.org/abs/2305.19302v3"]}
    )

    component_labels: Dict[str, List[List[Labels]]]

    def __init__(self, model_hypers: Dict, dataset_info: DatasetInfo) -> None:
        super().__init__()

        self.hypers = model_hypers
        self.dataset_info = dataset_info
        self.new_outputs = list(dataset_info.targets.keys())
        self.atomic_types = dataset_info.atomic_types

        self.requested_nl = NeighborListOptions(
            cutoff=self.hypers["edge"]["cutoff"]["radius"],
            full_list=True,
            strict=True,
        )

        self.linear = torch.nn.ModuleDict()

        self.outputs = {}
        self.key_labels: Dict[str, Labels] = {}
        self.component_labels: Dict[str, List[List[Labels]]] = {}
        self.property_labels: Dict[str, List[Labels]] = {}
        self.output_shapes: Dict[str, Dict[str, List[int]]] = {}
        self.output_info: Dict[str, Dict[str, str]] = {}

        # Initialize modules to compute descriptors
        # TODO: add selected keys
        self.descriptor_calculator = get_descriptor_calculator(
            model_hypers, dtype=torch.float64, atomic_types=self.atomic_types
        )
        self.descriptor_metadata = self.descriptor_calculator.compute_metadata(
            self.atomic_types
        )

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

        # Pretrain
        # TODO

        self.single_label = Labels.single()

    def _accumulate(self, systems: List[System], targets: Dict[str, TensorMap]):

        for target_name, target in targets.items():

            # Create the XTX and XTY blocks
            if target_name not in self._XTX:
                self._XTX[target_name] = {}
                self._XTY[target_name] = {}

            # Compute X
            features = self.descriptor_calculator(systems)

            for key, block in target.items():

                # Get the current block features
                X = features[key].values
                n_samples, n_components, n_properties = X.shape
                X = X.reshape(n_samples * n_components, n_properties)

                # Get the target block values
                values = block.values.reshape(n_samples * n_components, -1)

                is_invariant = key["o3_lambda"] == 0 and key["o3_sigma"] == 1
                if is_invariant:
                    X = torch.hstack(
                        [X, torch.ones((X.shape[0], 1), dtype=X.dtype, device=X.device)]
                    )

                # Compute a sparse XTX
                self._XTX[target_name][key].values[:] += X.T @ X

                # Compute a sparse XTY
                XTY = torch.einsum("sZ,sP->ZP", X, values)
                self._XTY[target_name][key].values[:] += XTY

    def pretrain(self, dataloader):

        # Initialize the XTX and XTY TensorMaps
        self._XTX: Dict[str, TensorMap] = {}
        self._XTY: Dict[str, TensorMap] = {}
        for target_name in self.dataset_info.targets:
            XTX_blocks: List[TensorBlock] = []
            XTY_blocks: List[TensorBlock] = []
            for key in self.dataset_info.targets[target_name].layout.keys:

                size = self.descriptor_metadata[key].shape[-1]
                if key["o3_lambda"] == 0 and key["o3_sigma"] == 1:
                    size = size + 1

                # Create the XTX blocks
                XTX_blocks.append(
                    TensorBlock(
                        values=torch.zeros(size, size, dtype=torch.float64),
                        samples=Labels("feature_1", torch.arange(size).reshape(-1, 1)),
                        components=[],
                        properties=Labels(
                            "feature_2", torch.arange(size).reshape(-1, 1)
                        ),
                    )
                )

                # Create the XTY blocks
                XTY_blocks.append(
                    TensorBlock(
                        values=torch.zeros(
                            size,
                            self.dataset_info.targets[target_name]
                            .layout[key]
                            .shape[-1],
                            dtype=torch.float64,
                        ),
                        samples=Labels("feature_1", torch.arange(size).reshape(-1, 1)),
                        components=[],
                        properties=Labels(
                            "feature_2",
                            torch.arange(
                                self.dataset_info.targets[target_name]
                                .layout[key]
                                .shape[-1]
                            ).reshape(-1, 1),
                        ),
                    )
                )

            self._XTX[target_name] = TensorMap(
                blocks=XTX_blocks,
                keys=self.dataset_info.targets[target_name].layout.keys,
            )
            self._XTY[target_name] = TensorMap(
                blocks=XTY_blocks,
                keys=self.dataset_info.targets[target_name].layout.keys,
            )

        # Accumulate XTX and XTY
        for batch in dataloader:
            systems, targets = batch
            self._accumulate(
                systems,
                {
                    target_name: targets[target_name]
                    for target_name in self.dataset_info.targets
                },
            )

        # Fit
        for target_name in self.dataset_info.targets:

            for key in self._XTX[target_name].keys:

                XTX = self._XTX[target_name][key].values
                XTY = self._XTY[target_name][key].values

                # Solve the linear system
                biases = None
                weights = _solve_linear_system(XTX, XTY)

                if key["o3_lambda"] == 0 and key["o3_sigma"] == 1:
                    biases = weights[:, -1]
                    weights = weights[:, :-1]

                with torch.no_grad():
                    model = self.linear[target_name].module_map.get_module(key)
                    model.weight.copy_(weights)
                    if model.bias is not None:
                        assert biases is not None
                        model.bias.copy_(biases)
                    else:
                        assert biases is None

    def _pretrain(self, dataloader):
        # TODO: delete when sure the pretrain method is 100% correct
        import metatensor.torch as mts

        features = []
        all_targets = {target_name: [] for target_name in self.dataset_info.targets}
        for batch in dataloader:
            systems, targets = batch
            features.append(self.descriptor_calculator(systems))
            for target_name in targets:
                all_targets[target_name].append(targets[target_name])
        features = mts.join(
            features,
            axis="samples",
            remove_tensor_name=True,
            different_keys="union",
        )

        for k in all_targets:
            all_targets[k] = mts.join(
                all_targets[k],
                axis="samples",
                remove_tensor_name=True,
                different_keys="union",
            )

        all_ridges = {}
        for target_name in all_targets:
            target = all_targets[target_name]
            all_ridges[target_name] = []
            for k in target.keys:
                X = features.block(k).values
                y = target.block(k).values
                n_samples, n_components, n_properties = X.shape
                X = X.reshape(n_samples * n_components, n_properties)
                y = y.reshape(n_samples * n_components, -1)
                ridge = RidgeCV(
                    alphas=np.logspace(-5, 5, 200),
                    fit_intercept=int(k["o3_lambda"]) == 0 and int(k["o3_sigma"]) == 1,
                )
                ridge.fit(X, y)
                all_ridges[target_name].append(ridge)

            self._apply_weights(all_ridges, target.keys)

    def _apply_weights(self, all_ridges: List[RidgeCV], keys) -> None:
        # TODO: delete when sure the pretrain method is 100% correct
        with torch.no_grad():
            for target_name in self.linear:
                module_map = self.linear[target_name].module_map
                ridges = all_ridges[target_name]
                for k, ridge in zip(keys, ridges):
                    model = module_map.get_module(k)
                    weights = torch.from_numpy(ridge.coef_)
                    if len(weights.shape) == 1:
                        weights = weights.unsqueeze(0)

                    model.weight.copy_(weights)
                    if model.bias is not None:
                        model.bias.copy_(torch.from_numpy(ridge.intercept_))

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs

    def restart(self, dataset_info: DatasetInfo) -> "MLElec":
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
                "The MLElec model does not support adding new atomic types."
            )

        # register new outputs as new last layers
        for target_name, target in new_targets.items():
            self._add_output(target_name, target)

        self.dataset_info = merged_info

        return self

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

        # system_indices = torch.concatenate(
        #     [
        #         torch.full(
        #             (len(system),),
        #             i_system,
        #             device=device,
        #         )
        #         for i_system, system in enumerate(systems)
        #     ],
        # )

        # Compute descriptors
        features = self.descriptor_calculator(systems)

        return_dict: Dict[str, TensorMap] = {}

        # Apply linear layers
        atomic_properties_tmap_dict: Dict[str, TensorMap] = {}
        for output_name, linear in self.linear.items():

            # Drop unused feature blocks
            blocks_to_discard = []
            for k in features.keys:
                if k not in self.dataset_info.targets[output_name].layout.keys:
                    blocks_to_discard.append(k.values)
            if len(blocks_to_discard) > 0:
                _features = metatensor.torch.drop_blocks(
                    features,
                    Labels(features.keys.names, torch.stack(blocks_to_discard)),
                )
            else:
                _features = features

            # Apply linear layer
            atomic_properties_tmap_dict[output_name] = linear(_features)

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

        return return_dict

    def requested_neighbor_lists(
        self,
    ) -> List[NeighborListOptions]:
        return [self.requested_nl]

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path]) -> "MLElec":
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
        # model.additive_models[0].sync_tensor_maps()

        return model

    def export(self, metadata: Optional[ModelMetadata] = None) -> AtomisticModel:
        dtype = next(self.parameters()).dtype
        if dtype not in self.__supported_dtypes__:
            raise ValueError(f"unsupported dtype {dtype} for MLElec")

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
            metadata = self.__default_metadata__
        else:
            merge_metadata(self.__default_metadata__, metadata)

        return AtomisticModel(self.eval(), metadata, capabilities)

    def _add_output(self, target_name: str, target_info: TargetInfo) -> None:
        """
        For the given target (i.e. TensorMap) whose name is ``target_name`` and info
        stored in ``target_info``, performs the following:

            - creates ModelOutput objects for the target and last layer features
            - stores the target metadata
            - stores the target output block shapes
            - stores extra useful output info
            - initializes the head module (MLP or identity)
            - initializes the output layers (linear)
        """

        # create ModelOutput objects for both the target and the last layer features
        self.outputs[target_name] = ModelOutput(
            quantity=target_info.quantity,
            unit=target_info.unit,
            per_atom=True,
        )

        # ll_features_name = (
        #     f"mtt::aux::{target_name.replace('mtt::', '')}_last_layer_features"
        # )
        # self.outputs[ll_features_name] = ModelOutput(per_atom=True)
        # assert self.outputs[ll_features_name].per_atom is True

        # store the target metadata
        self.key_labels[target_name] = target_info.layout.keys
        self.component_labels[target_name] = [
            block.components for block in target_info.layout.blocks()
        ]
        self.property_labels[target_name] = [
            block.properties for block in target_info.layout.blocks()
        ]

        # store the target output block shapes
        self.output_shapes[target_name] = {}
        for key, block in target_info.layout.items():
            # build the hashable str version of the key
            dict_key = target_name
            for n, k in zip(key.names, key.values):
                dict_key += f"_{n}_{int(k)}"
            # store the output shape indexed by the str key
            self.output_shapes[target_name][dict_key] = [
                len(comp.values) for comp in block.components
            ] + [len(block.properties.values)]

        # store some extra output info
        self.output_info[target_name] = {
            "target_type": target_info.target_type,
            "sample_kind": target_info.sample_kind,
            "symmetrized": (
                "true" if "s2_pi" in target_info.layout.keys.names else "false"
            ),
        }

        in_keys = target_info.layout.keys
        in_features = [self.descriptor_metadata[k].values.shape[-1] for k in in_keys]
        out_features = [b.values.shape[-1] for b in target_info.layout]
        out_properties = [b.properties for b in target_info.layout]
        self.linear[target_name] = EquivariantLinear(
            in_keys,
            in_features,
            out_features=out_features,
            out_properties=out_properties,
        )

        # # build the output layers. These transform last layer features into the output
        # self.last_layers[target_name] = torch.nn.ModuleDict(
        #     {
        #         key: torch.nn.Linear(
        #             self.hypers["d_pet"],
        #             prod(shape),
        #             bias=False,
        #         )
        #         for key, shape in self.output_shapes[target_name].items()
        #     }
        # )


def _get_output_block_slice_key(keys: Labels, key_i: int) -> str:
    if "center_type" in keys.names:
        slice_key = f"{keys.column('center_type')[key_i]}"
    else:
        assert "first_atom_type" in keys.names and "second_atom_type" in keys.names

        if "s2_pi" in keys.names:
            slice_key = (
                f"{keys.column('s2_pi')[key_i]}_"
                f"{keys.column('first_atom_type')[key_i]}_"
                f"{keys.column('second_atom_type')[key_i]}"
            )
        else:
            slice_key = (
                f"{keys.column('first_atom_type')[key_i]}_"
                f"{keys.column('second_atom_type')[key_i]}"
            )

    return slice_key


def _solve_linear_system(compf_t_at_compf, compf_t_at_targets) -> torch.Tensor:
    trace_magnitude = float(torch.diag(compf_t_at_compf).abs().mean())
    regularizer = 1e-3 * trace_magnitude
    if regularizer > 0.0:
        solution = torch.linalg.solve(
            compf_t_at_compf
            + regularizer
            * torch.eye(
                compf_t_at_compf.shape[1],
                dtype=compf_t_at_compf.dtype,
                device=compf_t_at_compf.device,
            ),
            compf_t_at_targets,
        ).T
    else:
        solution = torch.zeros(
            compf_t_at_targets.shape[1],
            compf_t_at_compf.shape[1],
            dtype=compf_t_at_compf.dtype,
            device=compf_t_at_compf.device,
        )
    return solution
