import metatensor.torch
from metatensor.torch import TensorMap

import torch
from typing import Dict, Optional

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
            loss += gradient_weight * self.loss(tensor_map_1.block().gradient(gradient_name).values, tensor_map_2.block().gradient(gradient_name).values)

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
