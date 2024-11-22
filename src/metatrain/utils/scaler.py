from typing import Dict, List, Union

import metatensor.torch
import numpy as np
import torch
from metatensor.torch import TensorMap
from metatensor.torch.atomistic import ModelOutput

from .additive import remove_additive
from .data import Dataset, DatasetInfo, TargetInfo, get_all_targets
from .jsonschema import validate


class Scaler(torch.nn.Module):
    """A class that scales the targets of regression problems.

    :param model_hypers: A dictionary of model hyperparameters. The paramater is ignored
        and is only present to be consistent with the general model API.
    :param dataset_info: An object containing information about the dataset, including
        target quantities and atomic types.

    :raises ValueError: If any target quantity in the dataset info is not an energy-like
        quantity.
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
        self.register_buffer("scales", torch.ones((0,), dtype=torch.float64))
        self.output_name_to_output_index: Dict[str, int] = {}
        for target_name, target_info in self.dataset_info.targets.items():
            self._add_output(target_name, target_info)

    def train_model(
        self,
        datasets: List[Union[Dataset, torch.utils.data.Subset]],
        additive_models: List[torch.nn.Module],
    ) -> None:
        """Calculate the scaling weights for all the targets in the datasets.

        :param datasets: Dataset(s) to calculate the scaling weights for.
        :param additive_models: Additive models to be removed from the targets
            before calculating the statistics.

        :raises ValueError: If the provided datasets contain unknown targets.
        """
        if not isinstance(datasets, list):
            datasets = [datasets]

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

                    for additive_model in additive_models:
                        target_info_dict = {target_key: self.new_targets[target_key]}
                        targets = remove_additive(
                            systems,
                            targets,
                            additive_model,
                            target_info_dict,
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
        """Scales all the targets in the outputs dictionary.

        :param outputs: A dictionary of target quantities and their values
            to be scaled.

        :raises ValueError: If an output is missing from the internal scales.
        """

        for output_name in outputs:
            if output_name.startswith("mtt::aux::"):
                continue
            if output_name not in self.outputs.keys():
                raise ValueError(
                    f"output key {output_name} is not supported by this scaler "
                    "model."
                )

        scaled_outputs: Dict[str, TensorMap] = {}
        for target_key, target in outputs.items():
            scale = float(
                self.scales[self.output_name_to_output_index[target_key]].item()
            )
            scaled_target = metatensor.torch.multiply(target, scale)
            scaled_outputs[target_key] = scaled_target

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
    """Remove the scaling from the targets.

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
