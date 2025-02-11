from typing import Dict, List, Union

import metatensor.torch
import numpy as np
import torch
from metatensor.torch import TensorMap
from metatensor.torch.atomistic import ModelOutput

from .additive import remove_additive
from .data import Dataset, DatasetInfo, TargetInfo, get_all_targets
from .jsonschema import validate
from .per_atom import average_by_num_atoms
from .transfer import systems_and_targets_to_device


class Scaler(torch.nn.Module):
    """
    A class that scales the targets of regression problems to unit standard
    deviation.

    In most cases, this should be used in conjunction with a composition model
    (that removes the multi-dimensional "mean" across the composition space) and/or
    other additive models. See the `train_model` method for more details.

    The scaling is performed per-atom, i.e., in cases where the targets are
    per-structure, the standard deviation is calculated on the targets divided by
    the number of atoms in each structure.

    :param model_hypers: A dictionary of model hyperparameters. The paramater is ignored
        and is only present to be consistent with the general model API.
    :param dataset_info: An object containing information about the dataset, including
        target quantities and atomic types.
    """

    outputs: Dict[str, ModelOutput]
    scales: torch.Tensor

    def __init__(self, model_hypers: Dict, dataset_info: DatasetInfo):
        super().__init__()

        # `model_hypers` should be an empty dictionary
        validate(
            instance=model_hypers,
            schema={"type": "object", "additionalProperties": False},
        )

        self.dataset_info = dataset_info

        self.new_targets: Dict[str, TargetInfo] = dataset_info.targets
        self.outputs: Dict[str, ModelOutput] = {}

        # Initially, the scales are empty. They will be expanded as new outputs
        # are registered with `_add_output`.
        self.register_buffer("scales", torch.ones((0,), dtype=torch.float64))
        self.output_name_to_output_index: Dict[str, int] = {}
        for target_name, target_info in self.dataset_info.targets.items():
            self._add_output(target_name, target_info)

    def train_model(
        self,
        datasets: List[Union[Dataset, torch.utils.data.Subset]],
        additive_models: List[torch.nn.Module],
        treat_as_additive: bool,
    ) -> None:
        """
        Calculate the scaling weights for all the targets in the datasets.

        :param datasets: Dataset(s) to calculate the scaling weights for.
        :param additive_models: Additive models to be removed from the targets
            before calculating the statistics.
        :param treat_as_additive: If True, all per-structure targets (i.e. those that)
            do not contain an ``atom`` label name, are treated as additive.

        :raises ValueError: If the provided datasets contain targets unknown
            to the scaler or if the targets are not treated as additive.
        """
        if not treat_as_additive:
            raise ValueError(
                "The Scaler class can currently only be trained by treating targets "
                "as additive."
            )

        if not isinstance(datasets, list):
            datasets = [datasets]

        device = self.scales.device

        # Fill the scales for each "new" target (i.e. those that do not already
        # have scales from a previous training run)
        for target_key in self.new_targets:
            datasets_with_target = []
            for dataset in datasets:
                if target_key in get_all_targets(dataset):
                    datasets_with_target.append(dataset)
            if len(datasets_with_target) == 0:
                raise ValueError(
                    f"Target {target_key} in the model's new capabilities is not "
                    "present in any of the training datasets."
                )

            sum_of_squared_targets = 0.0
            total_num_elements = 0
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
                            systems,
                            targets,
                            additive_model,
                            target_info_dict,
                        )

                    # calculate standard deviations on per-atom quantities
                    targets = average_by_num_atoms(
                        targets,
                        systems,
                        per_structure_keys=[],
                    )

                    target_info = self.new_targets[target_key]
                    if (
                        target_info.quantity == "energy"
                        and "positions" in target_info.gradients
                    ):
                        # special case: here we want to scale with respect to the forces
                        # rather than the energies
                        sum_of_squared_targets += torch.sum(
                            targets[target_key].block().gradient("positions").values
                            ** 2
                        ).item()
                        total_num_elements += (
                            targets[target_key]
                            .block()
                            .gradient("positions")
                            .values.numel()
                        )
                    else:
                        sum_of_squared_targets += sum(
                            torch.sum(block.values**2).item()
                            for block in targets[target_key].blocks()
                        )
                        total_num_elements += sum(
                            block.values.numel()
                            for block in targets[target_key].blocks()
                        )

            self.scales[self.output_name_to_output_index[target_key]] = np.sqrt(
                sum_of_squared_targets / total_num_elements
            )

    def restart(self, dataset_info: DatasetInfo) -> "Scaler":
        # merge old and new dataset info
        merged_info = self.dataset_info.union(dataset_info)

        self.new_targets = {
            key: value
            for key, value in merged_info.targets.items()
            if key not in self.dataset_info.targets
        }

        # register new outputs
        for target_name, target in self.new_targets.items():
            self._add_output(target_name, target)

        self.dataset_info = merged_info

        return self

    def forward(
        self,
        outputs: Dict[str, TensorMap],
    ) -> Dict[str, TensorMap]:
        """
        Scales all the targets in the outputs dictionary back to their
        original scale.

        :param outputs: A dictionary of target quantities and their values
            to be scaled.

        :raises ValueError: If an output does not have a corresponding
            scale in the scaler model.
        """
        scaled_outputs: Dict[str, TensorMap] = {}
        for target_key, target in outputs.items():
            if target_key in self.outputs:
                scale = float(
                    self.scales[self.output_name_to_output_index[target_key]].item()
                )
                scaled_target = metatensor.torch.multiply(target, scale)
                scaled_outputs[target_key] = scaled_target
            else:
                scaled_outputs[target_key] = target

        return scaled_outputs

    def _add_output(self, target_name: str, target_info: TargetInfo) -> None:
        self.outputs[target_name] = ModelOutput(
            quantity=target_info.quantity,
            unit=target_info.unit,
            per_atom=True,
        )

        self.scales = torch.cat(
            [self.scales, torch.tensor([1.0], dtype=self.scales.dtype)]
        )
        self.output_name_to_output_index[target_name] = len(self.scales) - 1

    def get_scales_dict(self) -> Dict[str, float]:
        """
        Return a dictionary with the scales for each output and output gradient.

        :return: A dictionary with the scales for each output and output gradient.
            These correspond to the standard deviation of the targets in the
            original dataset. The scales for each output gradient are the same
            as the corresponding output.
        """

        scales_dict = {
            output_name: self.scales[output_index].item()
            for output_name, output_index in self.output_name_to_output_index.items()
        }
        # Add gradients if present. They have the same scale as the corresponding output
        for output_name in list(scales_dict.keys()):
            gradient_names_for_output = self.dataset_info.targets[output_name].gradients
            for gradient_name in gradient_names_for_output:
                scales_dict[output_name + "_" + gradient_name + "_gradients"] = (
                    scales_dict[output_name]
                )
        return scales_dict


def remove_scale(
    targets: Dict[str, TensorMap],
    scaler: Scaler,
):
    """
    Scale all targets to a standard deviation of one.

    :param targets: Dictionary containing the targets to be scaled.
    :param scaler: The scaler used to scale the targets.
    """
    scaled_targets = {}
    for target_key in targets.keys():
        scale = float(
            scaler.scales[scaler.output_name_to_output_index[target_key]].item()
        )
        scaled_targets[target_key] = metatensor.torch.multiply(
            targets[target_key], 1.0 / scale
        )

    return scaled_targets
