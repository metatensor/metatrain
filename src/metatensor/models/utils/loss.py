import metatensor.torch
from metatensor.torch import TensorMap

from rascaline.torch.system import System

import torch
from typing import Dict, List, Optional

from .output_gradient import compute_gradient

# This file defines losses for metatensor models.


class TensorMapLoss:
    """
    A loss function that operates on two `metatensor.torch.TensorMap`s.
    
    The loss is computed as the sum of the loss on the block values and
    the loss on the gradients, with weights specified at initialization.

    This loss function assumes that all the gradients declared at
    initialization are present in both TensorMaps.
    """

    def __init__(
            self,
            reduction: str = "mean",
            weight: float = 1.0,
            gradient_weights: Optional[Dict[str, float]] = {},
        ):
        self.loss = torch.nn.MSELoss(reduction=reduction)
        self.weight = weight
        self.gradient_weights = gradient_weights

    def __call__(self, tensor_map_1: TensorMap, tensor_map_2: TensorMap) -> torch.Tensor:
        # Assert that the two have the same metadata:
        assert metatensor.torch.equal_metadata(tensor_map_1, tensor_map_2)

        # If the two TensorMaps have different symmetry keys:
        if len(tensor_map_1) != 1:
            raise NotImplementedError("TensorMapLoss does not yet support multiple symmetry keys.")

        # Compute the loss:
        loss = torch.zeros((), dtype=tensor_map_1.block().values.dtype, device=tensor_map_1.block().values.device)
        loss += self.weight * self.loss(tensor_map_1.block().values, tensor_map_2.block().values)
        for gradient_name, gradient_weight in self.gradient_weights.items():
            loss += gradient_weight * self.loss(tensor_map_1.gradient(gradient_name).values, tensor_map_2.gradient(gradient_name).values)

        return loss


class TensorMapDictLoss:
    """
    A loss function that operates on two `Dict[str, metatensor.torch.TensorMap]`.

    At initialization, the user specifies a list of keys to use for the loss,
    along with a weight for each key (as well as gradient weights).

    The loss is then computed as a weighted sum. Any keys that are not present
    in the dictionaries are ignored.
    """

    def __init__(
            self,
            weights: Dict[str, Dict[str, float]],
            reduction: str = "mean",
        ):
        self.losses = {}
        for key, weight in weights.items():
            # Remove the value weight from the gradient weights and store it separately:
            value_weight = weight.pop("values")  
            # Define the loss relative to this key:
            self.losses[key] = TensorMapLoss(reduction=reduction, weight=value_weight, gradient_weights=weight)

    def __call__(self, tensor_map_dict_1: Dict[str, TensorMap], tensor_map_dict_2: Dict[str, TensorMap]) -> torch.Tensor:
        # Assert that the two have the keys:
        assert set(tensor_map_dict_1.keys()) == set(tensor_map_dict_2.keys())

        # Initialize the loss:
        first_values = next(iter(tensor_map_dict_1.values())).block(0).values
        loss = torch.zeros((), dtype=first_values.dtype, device=first_values.device)

        # Compute the loss:
        for key in tensor_map_dict_1.keys():
            loss += self.losses[key](tensor_map_dict_1[key], tensor_map_dict_2[key])

        return loss


def compute_model_loss(
    loss: TensorMapDictLoss,
    model: torch.nn.Module,
    systems: List[System],
    targets: Dict[str, TensorMap],
):
    """
    Compute the loss of a model on a set of targets.

    This function assumes that the model returns a dictionary of
    TensorMaps, with the same keys as the targets.
    """
    # Assert that all targets are within the model's capabilities:
    if not set(targets.keys()).issubset(model.capabilities.outputs.keys()):
        raise ValueError("Not all targets are within the model's capabilities.")

    # Find if there are any energy targets that require gradients:
    energy_targets = []
    energy_targets_that_require_position_gradients = []
    energy_targets_that_require_displacement_gradients = []
    for target_name in targets.keys():
        # Check if the target is an energy:
        if model.capabilities.outputs[target_name].quantity == "energy":
            energy_targets.append(target_name)
            # Check if the energy requires gradients:
            if targets[target_name].has_gradients("positions"):
                energy_targets_that_require_position_gradients.append(target_name)
            if targets[target_name].has_gradients("displacements"):
                energy_targets_that_require_displacement_gradients.append(target_name)
                
    if len(energy_targets_that_require_displacement_gradients) > 0:
        # TODO: raise an error if the systems do not have a cell
        # if not all([system.has_cell for system in systems]):
        #     raise ValueError("One or more systems does not have a cell.")
        displacements = [torch.eye(3, requires_grad=True, dtype=system.dtype, device=system.device) for system in systems]
        # Create new "displaced" systems:
        systems = [
            System(
                positions=system.positions @ displacement,
                cell=system.cell @ displacement,
                species=system.species,
            )
            for system, displacement in zip(systems, displacements)
        ]
    else:
        if len(energy_targets_that_require_position_gradients) > 0:
            # Set positions to require gradients:
            for system in systems:
                system.positions.requires_grad_(True)

    # Based on the keys of the targets, get the outputs of the model:
    raw_model_outputs = model(systems, targets.keys())

    for energy_target in energy_targets:
        # If the energy target requires gradients, compute them:
        target_requires_pos_gradients = energy_target in energy_targets_that_require_position_gradients
        target_requires_disp_gradients = energy_target in energy_targets_that_require_displacement_gradients
        if target_requires_pos_gradients and target_requires_disp_gradients:
            gradients = compute_gradient(
                raw_model_outputs[energy_target].block().values,
                [system.positions for system in systems] + displacements,
                is_training=True,
            )
            new_energy_tensor_map
        elif target_requires_pos_gradients:
            gradients = compute_gradient(
                raw_model_outputs[energy_target].block().values,
                [system.positions for system in systems],
                is_training=True,
            )
        elif target_requires_disp_gradients:
            gradients = compute_gradient(
                raw_model_outputs[energy_target].block().values,
                displacements,
                is_training=True,
            )
        else:
            pass
