import logging
from typing import Dict, List, Optional, Union

import metatensor.torch
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import ModelOutput, System

from ..data import Dataset, DatasetInfo, TargetInfo, get_all_targets, get_atomic_types
from ..jsonschema import validate
from ..transfer import systems_and_targets_to_device
from .remove import remove_additive


logger = logging.getLogger(__name__)


class CompositionModel(torch.nn.Module):
    """A simple model that calculates the contributions to scalar targets
    based on the stoichiometry in a system.

    :param model_hypers: A dictionary of model hyperparameters. The paramater is ignored
        and is only present to be consistent with the general model API.
    :param dataset_info: An object containing information about the dataset, including
        target quantities and atomic types.
    """

    all_layouts = Dict[str, TensorMap]
    weights: Dict[str, TensorMap]
    outputs: Dict[str, ModelOutput]

    def __init__(self, model_hypers: Dict, dataset_info: DatasetInfo):
        super().__init__()

        # `model_hypers` should be an empty dictionary
        validate(
            instance=model_hypers,
            schema={"type": "object", "additionalProperties": False},
        )

        self.dataset_info = dataset_info
        self.atomic_types = sorted(dataset_info.atomic_types)

        for target_name, target_info in dataset_info.targets.items():
            if not self.is_valid_target(target_name, target_info):
                raise ValueError(
                    f"Composition model does not support target quantity "
                    f"{target_info.quantity}. This is an architecture bug. "
                    "Please report this issue and help us improve!"
                )

        self.new_targets = {
            target_name: target_info
            for target_name, target_info in dataset_info.targets.items()
        }

        self.all_layouts = {}
        self.weights = {}
        self.outputs: Dict[str, ModelOutput] = {}
        for target_name, target_info in self.dataset_info.targets.items():
            self._add_output(target_name, target_info)

        # keeps track of dtype and device of the composition model
        self.register_buffer("dummy_buffer", torch.randn(1))

    def train_model(
        self,
        datasets: List[Union[Dataset, torch.utils.data.Subset]],
        additive_models: List[torch.nn.Module],
        fixed_weights: Optional[Dict[str, Dict[int, str]]] = None,
    ) -> None:
        """Train/fit the composition weights for the datasets.

        :param datasets: Dataset(s) to calculate the composition weights for.
        :param fixed_weights: Optional fixed weights to use for the composition model,
            for one or more target quantities.
        :param additive_models: Additive models to be removed from the targets
            before calculating the statistics.

        :raises ValueError: If the provided datasets contain unknown targets.
        :raises ValueError: If the provided datasets contain unknown atomic types.
        :raises RuntimeError: If the linear system to calculate the composition weights
            cannot be solved.
        """
        if not isinstance(datasets, list):
            datasets = [datasets]

        if fixed_weights is None:
            fixed_weights = {}

        additional_types = sorted(
            set(get_atomic_types(datasets)) - set(self.atomic_types)
        )
        if additional_types:
            raise ValueError(
                "Provided `datasets` contains unknown "
                f"atomic types {additional_types}. "
                f"Known types from initialization are {self.atomic_types}."
            )

        missing_types = sorted(set(self.atomic_types) - set(get_atomic_types(datasets)))
        if missing_types:
            logger.warning(
                f"Provided `datasets` do not contain atomic types {missing_types}. "
                f"Known types from initialization are {self.atomic_types}."
            )

        device = self.dummy_buffer.device

        # Fill the weights for each "new" target (i.e. those that do not already
        # have composition weights from a previous training run)
        for target_key in self.new_targets:
            if target_key in fixed_weights:
                if not self.dataset_info.targets[target_key].is_scalar:
                    raise ValueError(
                        "Fixed weights can only be provided for scalar targets. "
                        f"Target {target_key} is not scalar."
                    )
                if (
                    len(self.dataset_info.targets[target_key].layout.block().properties)
                    != 1
                ):
                    raise ValueError(
                        "Fixed weights can only be provided for targets with one "
                        f"property. Target {target_key} has more than one property."
                    )
                # The fixed weights are provided for this target. Use them:
                if not sorted(fixed_weights[target_key].keys()) == self.atomic_types:
                    raise ValueError(
                        f"Fixed weights for target {target_key} must contain all "
                        f"atomic types {self.atomic_types}."
                    )
                weights_tensor = torch.tensor(
                    [fixed_weights[target_key][i] for i in self.atomic_types],
                    dtype=self.dummy_buffer.dtype,
                ).reshape(-1, 1)
                self.weights[target_key] = TensorMap(
                    keys=Labels.single(),
                    blocks=[
                        TensorBlock(
                            values=weights_tensor,
                            samples=Labels(
                                names=["center_type"],
                                values=torch.tensor(
                                    self.atomic_types, dtype=torch.int, device=device
                                ).reshape(-1, 1),
                            ),
                            components=self.dataset_info.targets[target_key]
                            .layout.block()
                            .components,
                            properties=self.dataset_info.targets[target_key]
                            .layout.block()
                            .properties,
                        )
                    ],
                )
            else:
                datasets_with_target = []
                for dataset in datasets:
                    if target_key in get_all_targets(dataset):
                        datasets_with_target.append(dataset)
                if len(datasets_with_target) == 0:
                    # this is a possibility when transfer learning
                    logger.warning(
                        f"Target {target_key} in the model's new capabilities is not "
                        "present in any of the training datasets."
                    )
                    continue

                dtype = datasets[0][0]["system"].positions.dtype
                if dtype != torch.float64:
                    raise ValueError(
                        "The composition model only supports float64 during training. "
                        f"Got dtype: {dtype}."
                    )

                is_spherical = self.dataset_info.targets[target_key].is_spherical
                is_per_atom = self.dataset_info.targets[target_key].per_atom

                if is_spherical:
                    if is_per_atom:
                        self.weights[target_key] = (
                            self._get_composition_spherical_per_atom(
                                datasets_with_target,
                                target_key,
                                additive_models,
                                device,
                                dtype,
                            )
                        )
                    else:
                        self.weights[target_key] = (
                            self._get_composition_spherical_per_structure(
                                datasets_with_target,
                                target_key,
                                additive_models,
                                device,
                                dtype,
                            )
                        )
                else:
                    if is_per_atom:
                        self.weights[target_key] = (
                            self._get_composition_scalar_per_atom(
                                datasets_with_target,
                                target_key,
                                additive_models,
                                device,
                                dtype,
                            )
                        )
                    else:
                        self.weights[target_key] = (
                            self._get_composition_scalar_per_structure(
                                datasets_with_target,
                                target_key,
                                additive_models,
                                device,
                                dtype,
                            )
                        )

                # for dataset in datasets_with_target:
                #     for sample in dataset:
                #         systems = [sample["system"]]
                #         targets = {target_key: sample[target_key]}
                #         systems, targets = systems_and_targets_to_device(
                #             systems, targets, device
                #         )
                #         for additive_model in additive_models:
                #             target_info_dict = {
                #                 target_key: self.new_targets[target_key]
                #             }
                #             targets = remove_additive(  # remove other additive models
                #                 systems,
                #                 targets,
                #                 additive_model,
                #                 target_info_dict,
                #             )
                #         for j, t in enumerate(self.atomic_types):
                #             composition_features[system_index, j] = torch.sum(
                #                 systems[0].types == t
                #             )
                #         system_index += 1
                #         if self.dataset_info.targets[target_key].per_atom:
                #             if "center_type" not in targets[
                #                 target_key
                #             ].keys.names and targets[target_key].block(
                #                 0
                #             ).samples.names == [
                #                 "system",
                #                 "atom",
                #             ]:
                #                 # there is no center type, we need to add it
                #                 # and we will rely on the fact that per-atom targets
                #                 # should be in the same order as the atoms in the system
                #                 targets[target_key] = metatensor.torch.append_dimension(
                #                     targets[target_key],
                #                     "samples",
                #                     "center_type",
                #                     systems[0].types,
                #                 )
                #         # TODO: abstract even more for more complex targets?
                #         for key, block in targets[target_key].items():
                #             # `if key not in per_block_targets_list` doesn't work, so:
                #             matching_keys = [
                #                 k for k in per_block_targets_list if k == key
                #             ]
                #             assert len(matching_keys) <= 1
                #             if len(matching_keys) == 0:
                #                 per_block_targets_list[key] = [block]
                #             else:
                #                 per_block_targets_list[matching_keys[0]].append(block)

                # weight_blocks = []
                # for key, block_list in per_block_targets_list.items():
                #     # distinguish between spherical and scalar targets
                #     is_spherical = self.dataset_info.targets[target_key].is_spherical
                #     is_spherical_and_invariant = False
                #     if is_spherical:
                #         is_spherical_and_invariant = (
                #             int(key["o3_lambda"]) == 0 and int(key["o3_sigma"]) == 1
                #         )
                #     needs_unsqueeze = False
                #     if self.dataset_info.targets[target_key].is_spherical:  # spherical
                #         is_invariant = (
                #             int(key["o3_lambda"]) == 0 and int(key["o3_sigma"]) == 1
                #         )
                #         if is_invariant:
                #             # squeeze components dimension
                #             tensor_list = [b.values.squeeze(1) for b in block_list]
                #             needs_unsqueeze = True
                #         else:
                #             # we don't need the targets as we will set the composition
                #             # to zero
                #             tensor_list = None
                #     else:  # scalar
                #         tensor_list = [b.values for b in block_list]

                #     metadata_block = self.dataset_info.targets[target_key].layout.block(
                #         key
                #     )
                #     if is_spherical and not is_spherical_and_invariant:
                #         weights_tensor = torch.zeros(
                #             (
                #                 len(self.atomic_types),
                #                 len(metadata_block.components[0]),
                #                 len(metadata_block.properties),
                #             ),
                #             dtype=dtype,
                #             device=device,
                #         )
                #     else:
                #         if self.dataset_info.targets[target_key].per_atom:
                #             # HACK: metatensor.join doesn't work on single blocks;
                #             # create TensorMaps, join, and then extract the joined block
                #             joined_blocks = metatensor.torch.join(
                #                 [
                #                     TensorMap(
                #                         keys=Labels.single(),
                #                         blocks=[b],
                #                     )
                #                     for b in block_list
                #                 ],
                #                 axis="samples",
                #                 remove_tensor_name=True,
                #             ).block()
                #             # This code doesn't work because mean_over_samples_block
                #             # actually does a sum... TODO: change for next release
                #             # weights_tensor = (
                #             #     metatensor.torch.sort_block(
                #             #         metatensor.torch.mean_over_samples_block(
                #             #             joined_blocks,
                #             #             [
                #             #                 n
                #             #                 for n in joined_blocks.samples.names
                #             #                 if n != "center_type"
                #             #             ],
                #             #         )
                #             #     )
                #             #     .values
                #             # )
                #             weights_tensor = torch.empty(
                #                 len(self.atomic_types), len(metadata_block.properties)
                #             )
                #             for i_type, atomic_type in enumerate(self.atomic_types):
                #                 mask = (
                #                     joined_blocks.samples.column("center_type")
                #                     == atomic_type
                #                 )
                #                 weights_tensor[i_type] = joined_blocks.values[
                #                     mask
                #                 ].mean(dim=0)
                #         else:
                #             # concatenate samples, for each block
                #             all_targets = torch.concatenate(tensor_list)
                #             weights_tensor = _solve_linear_system(
                #                 composition_features, all_targets
                #             )
                #     if needs_unsqueeze:  # scalar invariant, needs extra dimension
                #         weights_tensor = weights_tensor.unsqueeze(1)
                #     weight_blocks.append(
                #         TensorBlock(
                #             values=weights_tensor,
                #             samples=Labels(
                #                 ["center_type"],
                #                 values=torch.tensor(
                #                     self.atomic_types, dtype=torch.int, device=device
                #                 ).reshape(-1, 1),
                #             ),
                #             components=[
                #                 c.to(device) for c in metadata_block.components
                #             ],
                #             properties=metadata_block.properties.to(device),
                #         )
                #     )
                # self.weights[target_key] = TensorMap(
                #     keys=self.dataset_info.targets[target_key].layout.keys.to(device),
                #     blocks=weight_blocks,
                # )

    def restart(self, dataset_info: DatasetInfo) -> "CompositionModel":
        """Restart the model with a new dataset info.

        :param dataset_info: New dataset information to be used.
        """
        for target_name, target_info in dataset_info.targets.items():
            if not self.is_valid_target(target_name, target_info):
                raise ValueError(
                    f"Composition model does not support target "
                    f"{target_name}. This is an architecture bug. "
                    "Please report this issue and help us improve!"
                )

        # merge old and new dataset info
        merged_info = self.dataset_info.union(dataset_info)
        new_atomic_types = [
            at for at in merged_info.atomic_types if at not in self.atomic_types
        ]

        if len(new_atomic_types) > 0:
            raise ValueError(
                f"New atomic types found in the dataset: {new_atomic_types}. "
                "The composition model does not support adding new atomic types."
            )

        self.new_targets = {
            target_name: target_info
            for target_name, target_info in merged_info.targets.items()
            if target_name not in self.dataset_info.targets
        }

        # register new outputs
        for target_name, target in self.new_targets.items():
            self._add_output(target_name, target)

        self.dataset_info = merged_info

        return self

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        """Compute the targets for each system based on the composition weights.

        :param systems: List of systems to calculate the energy.
        :param outputs: Dictionary containing the model outputs.
        :param selected_atoms: Optional selection of atoms for which to compute the
            predictions.
        :returns: A dictionary with the computed predictions for each system.

        :raises ValueError: If no weights have been computed or if `outputs` keys
            contain unsupported keys.
        """
        dtype = systems[0].positions.dtype
        device = systems[0].positions.device

        # move weights (TensorMaps can't be treated as buffers for now)
        self._move_weights_to_device_and_dtype(device, dtype)

        for output_name in outputs:
            if output_name not in self.weights:
                raise ValueError(
                    f"output key {output_name} is not supported by this composition "
                    "model."
                )

        # Note: atomic types are not checked. At training time, the composition model
        # is initialized with the correct types. At inference time, the checks are
        # performed by MetatensorAtomisticModel.

        # create sample labels
        sample_values_list = []
        for i_system, system in enumerate(systems):
            system_column = torch.full(
                (len(system),), i_system, dtype=torch.int, device=device
            )
            atom_column = torch.arange(len(system), device=device)
            samples_values_single_system = torch.stack(
                [system_column, atom_column], dim=1
            )
            sample_values_list.append(samples_values_single_system)
        sample_values = torch.concatenate(sample_values_list)
        sample_labels = Labels(["system", "atom"], sample_values)

        # concatenate all types for all structures
        concatenated_types = torch.concatenate([system.types for system in systems])

        # Compute the output for each system by adding the composition weights times
        # number of atoms per atomic type.
        composition_result_dict: Dict[str, TensorMap] = {}
        for output_name, output_options in outputs.items():
            blocks: List[TensorBlock] = []
            if "center_type" in self.weights[output_name].keys.names:
                for key in self.all_layouts[output_name].keys:
                    if key in self.weights[output_name].keys:
                        weight_block = self.weights[output_name].block(key)
                        center_type = int(key["center_type"])
                        center_type_mask = concatenated_types == center_type
                        weights_tensor = weight_block.values
                        composition_values_per_atom = weights_tensor.expand(
                            [int(torch.sum(center_type_mask))] + [-1 for _ in weight_block.shape[1:]]
                        )
                        blocks.append(
                            TensorBlock(
                                values=composition_values_per_atom,
                                samples=Labels(
                                    sample_labels.names,
                                    sample_labels.values[center_type_mask],
                                ),
                                components=weight_block.components,
                                properties=weight_block.properties,
                            )
                        )
                    else:
                        center_type = int(key["center_type"])
                        center_type_mask = concatenated_types == center_type
                        blocks.append(
                            TensorBlock(
                                values=torch.zeros(
                                    [int(torch.sum(center_type_mask))]
                                    + self.all_layouts[output_name].block(key).shape[1:],
                                    dtype=dtype,
                                    device=device,
                                ),
                                samples=Labels(
                                    sample_labels.names,
                                    sample_labels.values[center_type_mask],
                                ),
                                components=self.all_layouts[output_name].block(key).components,
                                properties=self.all_layouts[output_name].block(key).properties,
                            )
                        )
            else:
                for key in self.all_layouts[output_name].keys:
                    if key in self.weights[output_name].keys:
                        weight_block = self.weights[output_name].block(key)
                        weights_tensor = weight_block.values
                        composition_values_per_atom = torch.empty(
                            [len(concatenated_types)] + weight_block.shape[1:],
                            dtype=dtype,
                            device=device,
                        )
                        for i_type, atomic_type in enumerate(self.atomic_types):
                            composition_values_per_atom[
                                concatenated_types == atomic_type
                            ] = weights_tensor[i_type]
                        blocks.append(
                            TensorBlock(
                                values=composition_values_per_atom,
                                samples=sample_labels,
                                components=weight_block.components,
                                properties=weight_block.properties,
                            )
                        )
                    else:  # spherical non-invariant target
                        blocks.append(
                            TensorBlock(
                                values=torch.zeros(
                                    [len(concatenated_types)]
                                    + self.all_layouts[output_name].block(key).shape[1:],
                                    dtype=dtype,
                                    device=device,
                                ),
                                samples=sample_labels,
                                components=self.all_layouts[output_name].block(key).components,
                                properties=self.all_layouts[output_name].block(key).properties,
                            )
                        )
            composition_result_dict[output_name] = TensorMap(
                keys=self.all_layouts[output_name].keys,
                blocks=blocks,
            )

            # apply selected_atoms to the composition if needed
            if selected_atoms is not None:
                composition_result_dict[output_name] = metatensor.torch.slice(
                    composition_result_dict[output_name], "samples", selected_atoms
                )

            if not output_options.per_atom:  # sum over atoms if needed
                composition_result_dict[output_name] = (
                    metatensor.torch.sum_over_samples(
                        composition_result_dict[output_name], sample_names="atom"
                    )
                )

        return composition_result_dict

    def _add_output(self, target_name: str, target_info: TargetInfo) -> None:
        self.all_layouts[target_name] = target_info.layout
        self.outputs[target_name] = ModelOutput(
            quantity=target_info.quantity,
            unit=target_info.unit,
            per_atom=True,
        )
        center_type_in_keys = "center_type" in target_info.layout.keys.names
        new_keys = Labels(
            target_info.layout.keys.names, target_info.layout.keys.values[target_info.layout.keys.select(
                Labels(["o3_lambda", "o3_sigma"], torch.tensor([[0, 1]]))
            )]
        ) if target_info.is_spherical else target_info.layout.keys
        if center_type_in_keys:
            self.weights[target_name] = TensorMap(
                keys=new_keys,
                blocks=[
                    TensorBlock(
                        values=torch.zeros(
                            ([1] + block.shape[1:]),
                            dtype=torch.float64,
                        ),
                        samples=Labels.single(),
                        components=block.components,
                        properties=block.properties,
                    )
                    for key, block in target_info.layout.items()
                    if (key["o3_lambda"] == 0 and key["o3_sigma"] == 1)
                ],
            )
        else:
            self.weights[target_name] = TensorMap(
                keys=new_keys,
                blocks=[
                    TensorBlock(
                        values=torch.zeros(
                            ([len(self.atomic_types)] + block.shape[1:]),
                            dtype=torch.float64,
                        ),
                        samples=Labels(
                            names=["center_type"],
                            values=torch.tensor(
                                self.atomic_types, dtype=torch.int
                            ).reshape(-1, 1),
                        ),
                        components=block.components,
                        properties=block.properties,
                    )
                    for key, block in target_info.layout.items()
                    if (key["o3_lambda"] == 0 and key["o3_sigma"] == 1)
                ],
            )

    def _move_weights_to_device_and_dtype(
        self, device: torch.device, dtype: torch.dtype
    ):
        if len(self.weights) != 0:
            if self.weights[list(self.weights.keys())[0]].device != device:
                self.weights = {k: v.to(device) for k, v in self.weights.items()}
            if self.weights[list(self.weights.keys())[0]].dtype != dtype:
                self.weights = {k: v.to(dtype) for k, v in self.weights.items()}
        if len(self.all_layouts) != 0:
            if self.all_layouts[list(self.all_layouts.keys())[0]].device != device:
                self.all_layouts = {k: v.to(device) for k, v in self.all_layouts.items()}
            if self.all_layouts[list(self.all_layouts.keys())[0]].dtype != dtype:
                self.all_layouts = {k: v.to(dtype) for k, v in self.all_layouts.items()}

    def _get_composition_spherical_per_atom(
        self,
        datasets_with_target: List[Union[Dataset, torch.utils.data.Subset]],
        target_key: str,
        additive_models: List[torch.nn.Module],
        device: torch.device,
        dtype: torch.dtype,
    ):
        metadata_tensor_map = self.dataset_info.targets[target_key].layout
        center_type_in_keys = "center_type" in metadata_tensor_map.keys.names

        # Initialize one accumulator per block (only invariant blocks)
        if center_type_in_keys:
            mean_accumulators = {
                tuple(int(k) for k in key.values): _MeanAccumulator(
                    shape=metadata_tensor_map.block(key).values.shape[1:],
                    device=device,
                    dtype=dtype,
                )
                for key in metadata_tensor_map.keys
                if (key["o3_lambda"] == 0 and key["o3_sigma"] == 1)
            }
        else:
            mean_accumulators = {
                tuple(int(k) for k in key.values) + (center_type): _MeanAccumulator(
                    shape=metadata_tensor_map.block(key).values.shape[1:],
                    device=device,
                    dtype=dtype,
                )
                for center_type in self.atomic_types
                for key in metadata_tensor_map.keys
                if (key["o3_lambda"] == 0 and key["o3_sigma"] == 1)
            }

        for dataset in datasets_with_target:
            for sample in dataset:
                systems = [sample["system"]]
                targets = {target_key: sample[target_key]}
                systems, targets = systems_and_targets_to_device(
                    systems, targets, device
                )
                for additive_model in additive_models:
                    target_info_dict = {target_key: self.new_targets[target_key]}
                    targets = remove_additive(
                        systems, targets, additive_model, target_info_dict
                    )
                for key, block in targets[target_key].items():
                    if key["o3_lambda"] == 0 and key["o3_sigma"] == 1:
                        # Two cases: with and without center_type
                        if center_type_in_keys:
                            mean_accumulators[tuple(int(k) for k in key.values)].add(
                                block.values
                            )
                        else:
                            for center_type in self.atomic_types:
                                mask = systems[0].types == center_type
                                mean_accumulators[
                                    tuple(int(k) for k in key.values) + (center_type,)
                                ].add(block.values[mask])

        composition_tensor_map = TensorMap(
            keys=Labels(
                names=metadata_tensor_map.keys.names,
                values=torch.stack(
                    [k.values for k in metadata_tensor_map.keys if (k["o3_lambda"] == 0 and k["o3_sigma"] == 1)]
                ).to(device),
            ),
            blocks=(
                [
                    TensorBlock(
                        values=mean_accumulators[tuple(int(k) for k in key.values)]
                        .return_result()
                        .reshape((1,) + metadata_tensor_map.block(key).values.shape[1:]),
                        samples=Labels.single().to(device),
                        components=[c.to(device) for c in metadata_tensor_map.block(key).components],
                        properties=self.dataset_info.targets[target_key]
                        .layout.block(key)
                        .properties.to(device),
                    )
                    for key in metadata_tensor_map.keys if (key["o3_lambda"] == 0 and key["o3_sigma"] == 1)
                ]
                if center_type_in_keys
                else [
                    TensorBlock(
                        values=torch.stack(
                            [
                                mean_accumulators[
                                    tuple(int(k) for k in key.values) + (center_type,)
                                ].return_result()
                                for center_type in self.atomic_types
                            ]
                        ),
                        samples=Labels(
                            names=["center_type"],
                            values=torch.tensor(
                                self.atomic_types, dtype=torch.int, device=device
                            ).reshape(-1, 1),
                        ),
                        components=[c.to(device) for c in metadata_tensor_map.block(key).components],
                        properties=self.dataset_info.targets[target_key]
                        .layout.block(key)
                        .properties.to(device),
                    )
                    for key in metadata_tensor_map.keys if (key["o3_lambda"] == 0 and key["o3_sigma"] == 1)
                ]
            ),
        )
        return composition_tensor_map

    def _get_composition_spherical_per_structure(
        self,
        datasets_with_target: List[Union[Dataset, torch.utils.data.Subset]],
        target_key: str,
        additive_models: List[torch.nn.Module],
        device: torch.device,
        dtype: torch.dtype,
    ):
        raise NotImplementedError()

    def _get_composition_scalar_per_atom(
        self,
        datasets_with_target: List[Union[Dataset, torch.utils.data.Subset]],
        target_key: str,
        additive_models: List[torch.nn.Module],
        device: torch.device,
        dtype: torch.dtype,
    ):
        raise NotImplementedError()

    def _get_composition_scalar_per_structure(
        self,
        datasets_with_target: List[Union[Dataset, torch.utils.data.Subset]],
        target_key: str,
        additive_models: List[torch.nn.Module],
        device: torch.device,
        dtype: torch.dtype,
    ):
        raise NotImplementedError()

    @staticmethod
    def is_valid_target(target_name: str, target_info: TargetInfo) -> bool:
        """Finds if a ``TargetInfo`` object is compatible with a composition model.

        :param target_info: The ``TargetInfo`` object to be checked.
        """
        # only scalars can have composition contributions
        if not target_info.is_scalar and not target_info.is_spherical:
            logger.debug(
                f"Composition model does not support target {target_name} "
                "since it is not either scalar or spherical."
            )
            return False
        if (
            target_info.is_spherical
            and len(target_info.layout.blocks({"o3_lambda": 0, "o3_sigma": 1})) == 0
        ):
            logger.debug(
                f"Composition model does not support spherical target {target_name} "
                "since it does not have any invariant blocks."
            )
            return False
        return True


class _MeanAccumulator:
    def __init__(self, shape: List[int], device: torch.device, dtype: torch.dtype):
        self.sum = torch.zeros(shape, dtype=dtype, device=device)
        self.count = 0

    def add(self, tensor: float):
        self.sum += torch.sum(tensor, dim=0)
        self.count += tensor.numel()

    def return_result(self) -> torch.Tensor:
        return self.sum / self.count


class _LinearSystemAccumulator:
    def __init__(
        self,
        feature_size: int,
        target_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.feat_t_at_feat = torch.zeros(
            feature_size, feature_size, dtype=dtype, device=device
        )
        self.feat_t_at_targets = torch.zeros(
            feature_size, target_size, dtype=dtype, device=device
        )

    def add(self, features: torch.Tensor, targets: torch.Tensor):
        self.feat_t_at_feat += features.T @ features
        self.feat_t_at_targets += features.T @ targets

    def return_result(self) -> torch.Tensor:
        trace_magnitude = float(torch.diag(self.compf_t_at_compf).abs().mean())
        regularizer = 1e-14 * trace_magnitude
        return torch.linalg.solve(
            self.feat_t_at_feat
            + regularizer
            * torch.eye(
                self.feat_t_at_feat.shape[0],
                dtype=self.feat_t_at_feat.dtype,
                device=self.feat_t_at_feat.device,
            ),
            self.feat_t_at_targets,
        )
