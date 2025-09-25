from typing import Dict, List, Union

import metatensor.torch as mts
import numpy as np
import torch
from metatensor.torch import TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System

from metatrain.utils.data import Dataset, DatasetInfo, TargetInfo, get_all_targets
from metatrain.utils.scaler import Scaler as BaseScaler
from metatrain.utils.transfer import batch_to


class Scaler(BaseScaler):
    def __init__(self, hypers: Dict, dataset_info: DatasetInfo):
        super().__init__(hypers, dataset_info)

        # check that the registered targets are "positions" and "momenta"
        expected_keys = {"positions", "momenta"}
        if set(self.output_name_to_output_index.keys()) != expected_keys:
            raise ValueError(
                f"Expected target keys to be {expected_keys}, "
                f"got {set(self.output_name_to_output_index.keys())}"
            )

        # register masses
        self.register_buffer(
            "masses",
            torch.full((max(self.dataset_info.atomic_types) + 1,), float("nan")),
        )

    def set_masses(self, masses: Dict[int, float]):
        for atomic_number, mass in masses.items():
            self.masses[atomic_number] = mass

    def train_model(
        self,
        datasets: List[Union[Dataset, torch.utils.data.Subset]],
        additive_models: List[torch.nn.Module],
        treat_as_additive: bool,
    ) -> None:
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

                    systems, targets, _ = batch_to(systems, targets, device=device)

                    # Calculate scaled positions and momenta before accumulating
                    # statistics
                    tensor_with_targets = targets[target_key].block().values
                    atomic_types = torch.concatenate(
                        [system.types for system in systems]
                    )
                    masses = self.masses[atomic_types]
                    if target_key == "positions":
                        tensor_with_targets = (
                            tensor_with_targets
                            - torch.concatenate(
                                [system.positions for system in systems]
                            ).unsqueeze(-1)
                        ) * torch.sqrt(masses[:, None, None])
                    elif target_key == "momenta":
                        tensor_with_targets = tensor_with_targets / torch.sqrt(
                            masses[:, None, None]
                        )
                    else:
                        raise ValueError(
                            "Scaler can only scale 'positions' and 'momenta',"
                            f" got {target_key}"
                        )

                    sum_of_squared_targets += torch.sum(tensor_with_targets**2).item()
                    total_num_elements += tensor_with_targets.numel()

            self.scales[self.output_name_to_output_index[target_key]] = np.sqrt(
                sum_of_squared_targets / total_num_elements
            )

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
                scaled_target = mts.multiply(target, scale)
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


def remove_scale(
    systems: List[System],
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
        # Special case: positions. In this case, the scaling needs to be applied to
        # the difference between positions in the targets and those in the systems
        if target_key == "positions":
            positions_before = torch.concatenate(
                [system.positions for system in systems]
            )
            positions_after = targets[target_key].block().values.squeeze(-1)
            diff_positions = positions_after - positions_before
            scaled_diff_positions = diff_positions / scale
            scaled_positions = positions_before + scaled_diff_positions
            scaled_targets[target_key] = TensorMap(
                keys=targets[target_key].keys,
                blocks=[
                    TensorBlock(
                        values=scaled_positions.unsqueeze(-1),
                        samples=targets[target_key].block().samples,
                        components=targets[target_key].block().components,
                        properties=targets[target_key].block().properties,
                    )
                ],
            )
        else:
            scaled_targets[target_key] = mts.multiply(targets[target_key], 1.0 / scale)

    return scaled_targets
