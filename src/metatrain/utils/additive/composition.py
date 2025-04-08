import logging
from typing import Dict, List, Optional, Union

import metatensor.torch
import torch
from metatensor.torch import Labels, LabelsEntry, TensorBlock, TensorMap
from metatensor.torch.atomistic import ModelOutput, System

from ..data import Dataset, DatasetInfo, TargetInfo, get_all_targets, get_atomic_types
from ..jsonschema import validate
from ..sum_over_atoms import sum_over_atoms
from ..transfer import systems_and_targets_to_device
from .remove import remove_additive


class CompositionModel(torch.nn.Module):
    """A simple model that calculates the contributions to scalar targets
    based on the stoichiometry in a system.

    :param model_hypers: A dictionary of model hyperparameters. The paramater is ignored
        and is only present to be consistent with the general model API.
    :param dataset_info: An object containing information about the dataset, including
        target quantities and atomic types.
    """

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

        self.register_buffer(
            "type_to_index", torch.empty(max(self.atomic_types) + 1, dtype=torch.long)
        )
        for i, atomic_type in enumerate(self.atomic_types):
            self.type_to_index[atomic_type] = i

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
            logging.warning(
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
                    device=device,
                ).reshape(-1, 1)
                self.weights[target_key] = TensorMap(
                    keys=Labels.single().to(device),
                    blocks=[
                        TensorBlock(
                            values=weights_tensor,
                            samples=Labels(
                                names=["center_type"],
                                values=torch.tensor(
                                    self.atomic_types, dtype=torch.int, device=device
                                ).reshape(-1, 1),
                            ),
                            components=[
                                c.to(device)
                                for c in self.dataset_info.targets[target_key]
                                .layout.block()
                                .components
                            ],
                            properties=self.dataset_info.targets[target_key]
                            .layout.block()
                            .properties.to(device),
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
                    logging.warning(
                        f"Target {target_key} in the model's new capabilities is not "
                        "present in any of the training datasets."
                    )
                    continue

                total_num_structures = sum(
                    [len(dataset) for dataset in datasets_with_target]
                )
                dtype = datasets[0][0]["system"].positions.dtype
                if dtype != torch.float64:
                    raise ValueError(
                        "The composition model only supports float64 during training. "
                        f"Got dtype: {dtype}."
                    )

                composition_features = torch.zeros(
                    (total_num_structures, len(self.atomic_types)),
                    dtype=dtype,
                    device=device,
                )
                system_index = 0
                per_block_targets_list: Dict[LabelsEntry, List[TensorBlock]] = {}
                for dataset in datasets_with_target:
                    for sample in dataset:
                        systems = [sample["system"]]
                        targets = {target_key: sample[target_key]}
                        systems, targets = systems_and_targets_to_device(
                            systems, targets, device
                        )
                        for additive_model in additive_models:
                            target_info_dict = {
                                target_key: self.new_targets[target_key]
                            }
                            targets = remove_additive(  # remove other additive models
                                systems,
                                targets,
                                additive_model,
                                target_info_dict,
                            )
                        for j, t in enumerate(self.atomic_types):
                            composition_features[system_index, j] = torch.sum(
                                systems[0].types == t
                            )
                        system_index += 1
                        if self.dataset_info.targets[target_key].per_atom:
                            # we need the center type in the samples to do
                            # mean_over_samples
                            if "center_type" in targets[target_key].keys.names:
                                # it's in the keys: move it to the samples
                                targets[target_key] = targets[
                                    target_key
                                ].keys_to_samples("center_type")
                            if targets[target_key].block(0).samples.names == [
                                "system",
                                "atom",
                            ]:
                                # there is no center type, we need to add it
                                # and we will rely on the fact that per-atom targets
                                # should be in the same order as the atoms in the system
                                targets[target_key] = metatensor.torch.append_dimension(
                                    targets[target_key],
                                    "samples",
                                    "center_type",
                                    systems[0].types,
                                )
                        # TODO: abstract even more for more complex targets?
                        for key, block in targets[target_key].items():
                            # `if key not in per_block_targets_list` doesn't work, so:
                            matching_keys = [
                                k for k in per_block_targets_list if k == key
                            ]
                            assert len(matching_keys) <= 1
                            if len(matching_keys) == 0:
                                per_block_targets_list[key] = [block]
                            else:
                                per_block_targets_list[matching_keys[0]].append(block)

                weight_blocks = []
                for key, block_list in per_block_targets_list.items():
                    # distinguish between spherical and scalar targets
                    needs_unsqueeze = False
                    if self.dataset_info.targets[target_key].is_spherical:  # spherical
                        is_invariant = (
                            int(key["o3_lambda"]) == 0 and int(key["o3_sigma"]) == 1
                        )
                        if is_invariant:
                            # squeeze components dimension
                            tensor_list = [b.values.squeeze(1) for b in block_list]
                            needs_unsqueeze = True
                        else:
                            # we don't need the targets as we will set the composition
                            # to zero
                            tensor_list = None
                    else:  # scalar
                        tensor_list = [b.values for b in block_list]

                    metadata_block = self.dataset_info.targets[target_key].layout.block(
                        key
                    )
                    if tensor_list is None:  # spherical non-invariant
                        weights_tensor = torch.zeros(
                            (
                                len(self.atomic_types),
                                len(metadata_block.components[0]),
                                len(metadata_block.properties),
                            ),
                            dtype=dtype,
                            device=device,
                        )
                    else:
                        if self.dataset_info.targets[target_key].per_atom:
                            # hack: metatensor.join doesn't work on single blocks;
                            # create TensorMaps, join, and then extract the joined block
                            joined_blocks = metatensor.torch.join(
                                [
                                    TensorMap(
                                        keys=Labels.single(),
                                        blocks=[b],
                                    )
                                    for b in block_list
                                ],
                                axis="samples",
                                remove_tensor_name=True,
                            ).block()
                            weights_tensor = metatensor.torch.sort_block(
                                metatensor.torch.mean_over_samples_block(
                                    joined_blocks,
                                    [
                                        n
                                        for n in joined_blocks.samples.names
                                        if n != "center_type"
                                    ],
                                )
                            ).values
                        else:
                            # concatenate samples, for each block
                            all_targets = torch.concatenate(tensor_list)
                            weights_tensor = _solve_linear_system(
                                composition_features, all_targets
                            )
                    if needs_unsqueeze:  # scalar invariant, needs extra dimension
                        weights_tensor = weights_tensor.unsqueeze(1)
                    weight_blocks.append(
                        TensorBlock(
                            values=weights_tensor.contiguous(),
                            # TODO: remove the .contiguous() when metatensor supports it
                            samples=Labels(
                                ["center_type"],
                                values=torch.tensor(
                                    self.atomic_types, dtype=torch.int, device=device
                                ).reshape(-1, 1),
                            ),
                            components=[
                                c.to(device) for c in metadata_block.components
                            ],
                            properties=metadata_block.properties.to(device),
                        )
                    )
                self.weights[target_key] = TensorMap(
                    keys=self.dataset_info.targets[target_key].layout.keys.to(device),
                    blocks=weight_blocks,
                )

            # make sure to update the weights buffer with the new weights
            self.register_buffer(
                target_key + "_composition_buffer",
                metatensor.torch.save_buffer(
                    self.weights[target_key].to("cpu", torch.float64)
                ).to(device),
            )

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
            for weight_key, weight_block in self.weights[output_name].items():
                weights_tensor = self.weights[output_name].block(weight_key).values
                composition_values_per_atom = weights_tensor[
                    self.type_to_index[concatenated_types]
                ]
                blocks.append(
                    TensorBlock(
                        values=composition_values_per_atom,
                        samples=sample_labels,
                        components=weight_block.components,
                        properties=weight_block.properties,
                    )
                )
            composition_result_dict[output_name] = TensorMap(
                keys=self.weights[output_name].keys,
                blocks=blocks,
            )

            # apply selected_atoms to the composition if needed
            if selected_atoms is not None:
                composition_result_dict[output_name] = metatensor.torch.slice(
                    composition_result_dict[output_name], "samples", selected_atoms
                )

            if not output_options.per_atom:  # sum over atoms if needed
                composition_result_dict[output_name] = sum_over_atoms(
                    composition_result_dict[output_name]
                )

        return composition_result_dict

    def _add_output(self, target_name: str, target_info: TargetInfo) -> None:
        self.outputs[target_name] = ModelOutput(
            quantity=target_info.quantity,
            unit=target_info.unit,
            per_atom=True,
        )
        self.weights[target_name] = TensorMap(
            keys=target_info.layout.keys,
            blocks=[
                TensorBlock(
                    values=torch.zeros(
                        ([len(self.atomic_types)] + b.shape[1:]),
                        dtype=torch.float64,
                    ),
                    samples=Labels(
                        names=["center_type"],
                        values=torch.tensor(self.atomic_types, dtype=torch.int).reshape(
                            -1, 1
                        ),
                    ),
                    components=b.components,
                    properties=b.properties,
                )
                for b in target_info.layout.blocks()
            ],
        )

        # register a buffer to store the weights; this is necessary because the weights
        # are TensorMaps and cannot be stored in the state_dict
        fake_weights = TensorMap(
            keys=self.dataset_info.targets[target_name].layout.keys,
            blocks=[
                TensorBlock(
                    values=torch.zeros(
                        (len(self.atomic_types),) + b.values.shape[1:],
                        dtype=torch.float64,
                    ),
                    samples=Labels(
                        names=["center_type"],
                        values=torch.tensor(self.atomic_types, dtype=torch.int).reshape(
                            -1, 1
                        ),
                    ),
                    components=b.components,
                    properties=b.properties,
                )
                for b in target_info.layout.blocks()
            ],
        )
        self.register_buffer(
            target_name + "_composition_buffer",
            metatensor.torch.save_buffer(fake_weights),
        )

    def _move_weights_to_device_and_dtype(
        self, device: torch.device, dtype: torch.dtype
    ):
        if len(self.weights) != 0:
            if self.weights[list(self.weights.keys())[0]].device != device:
                self.weights = {k: v.to(device) for k, v in self.weights.items()}
            if self.weights[list(self.weights.keys())[0]].dtype != dtype:
                self.weights = {k: v.to(dtype) for k, v in self.weights.items()}

    @staticmethod
    def is_valid_target(target_name: str, target_info: TargetInfo) -> bool:
        """Finds if a ``TargetInfo`` object is compatible with a composition model.

        :param target_info: The ``TargetInfo`` object to be checked.
        """
        # only scalars can have composition contributions
        if not target_info.is_scalar and not target_info.is_spherical:
            logging.debug(
                f"Composition model does not support target {target_name} "
                "since it is not either scalar or spherical."
            )
            return False
        if (
            target_info.is_spherical
            and len(target_info.layout.blocks({"o3_lambda": 0, "o3_sigma": 1})) == 0
        ):
            logging.debug(
                f"Composition model does not support spherical target {target_name} "
                "since it does not have any invariant blocks."
            )
            return False
        return True

    def sync_tensor_maps(self):
        # Reload the weights of the (old) targets, which are not stored in the model
        # state_dict, from the buffers
        for k in self.dataset_info.targets:
            self.weights[k] = metatensor.torch.load_buffer(
                self.__getattr__(k + "_composition_buffer")
            )


def _solve_linear_system(composition_features, all_targets) -> torch.Tensor:
    compf_t_at_compf = composition_features.T @ composition_features
    compf_t_at_targets = composition_features.T @ all_targets
    trace_magnitude = float(torch.diag(compf_t_at_compf).abs().mean())
    regularizer = 1e-14 * trace_magnitude
    return torch.linalg.solve(
        compf_t_at_compf
        + regularizer
        * torch.eye(
            composition_features.shape[1],
            dtype=composition_features.dtype,
            device=composition_features.device,
        ),
        compf_t_at_targets,
    )
