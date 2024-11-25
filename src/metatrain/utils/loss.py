from typing import Dict, Optional, Tuple, Union

import torch
from metatensor.torch import TensorMap
from omegaconf import DictConfig

from metatrain.utils.external_naming import to_internal_name


# This file defines losses for metatensor models.


class TensorMapLoss:
    """A loss function that operates on two ``metatensor.torch.TensorMap``.

    The loss is computed as the sum of the loss on the block values and
    the loss on the gradients, with weights specified at initialization.

    At the moment, this loss function assumes that all the gradients
    declared at initialization are present in both TensorMaps.

    :param reduction: The reduction to apply to the loss.
        See :py:class:`torch.nn.MSELoss`.
    :param weight: The weight to apply to the loss on the block values.
    :param gradient_weights: The weights to apply to the loss on the gradients.

    :returns: The loss as a zero-dimensional :py:class:`torch.Tensor`
        (with one entry).
    """

    def __init__(
        self,
        reduction: str = "sum",
        weight: float = 1.0,
        gradient_weights: Optional[Dict[str, float]] = None,
        type: Union[str, dict] = "mse",
    ):
        if gradient_weights is None:
            gradient_weights = {}

        losses = {}
        if type == "mse":
            losses["values"] = torch.nn.MSELoss(reduction=reduction)
            for key in gradient_weights.keys():
                losses[key] = torch.nn.MSELoss(reduction=reduction)
        elif type == "mae":
            losses["values"] = torch.nn.L1Loss(reduction=reduction)
            for key in gradient_weights.keys():
                losses[key] = torch.nn.L1Loss(reduction=reduction)
        elif isinstance(type, dict) and "huber" in type:
            # Huber loss
            deltas = type["huber"]["deltas"]
            losses["values"] = torch.nn.HuberLoss(
                reduction=reduction, delta=deltas["values"]
            )
            for key in gradient_weights.keys():
                losses[key] = torch.nn.HuberLoss(reduction=reduction, delta=deltas[key])
        else:
            raise ValueError(f"Unknown loss type: {type}")

        self.losses = losses
        self.weight = weight
        self.gradient_weights = gradient_weights

    def __call__(
        self, tensor_map_1: TensorMap, tensor_map_2: TensorMap
    ) -> Tuple[torch.Tensor, Dict[str, Tuple[float, int]]]:
        # Check that the two have the same metadata, except for the samples,
        # which can be different due to batching, but must have the same size:
        if tensor_map_1.keys != tensor_map_2.keys:
            raise ValueError(
                "TensorMapLoss requires the two TensorMaps to have the same keys."
            )
        for block_1, block_2 in zip(tensor_map_1.blocks(), tensor_map_2.blocks()):
            if block_1.properties != block_2.properties:
                raise ValueError(
                    "TensorMapLoss requires the two TensorMaps to have the same "
                    "properties."
                )
            if block_1.components != block_2.components:
                raise ValueError(
                    "TensorMapLoss requires the two TensorMaps to have the same "
                    "components."
                )
            if len(block_1.samples) != len(block_2.samples):
                raise ValueError(
                    "TensorMapLoss requires the two TensorMaps "
                    "to have the same number of samples."
                )
            for gradient_name in self.gradient_weights.keys():
                if len(block_1.gradient(gradient_name).samples) != len(
                    block_2.gradient(gradient_name).samples
                ):
                    raise ValueError(
                        "TensorMapLoss requires the two TensorMaps "
                        "to have the same number of gradient samples."
                    )
                if (
                    block_1.gradient(gradient_name).properties
                    != block_2.gradient(gradient_name).properties
                ):
                    raise ValueError(
                        "TensorMapLoss requires the two TensorMaps "
                        "to have the same gradient properties."
                    )
                if (
                    block_1.gradient(gradient_name).components
                    != block_2.gradient(gradient_name).components
                ):
                    raise ValueError(
                        "TensorMapLoss requires the two TensorMaps "
                        "to have the same gradient components."
                    )

        # Compute the loss:
        loss = torch.zeros(
            (),
            dtype=tensor_map_1.block(0).values.dtype,
            device=tensor_map_1.block(0).values.device,
        )

        for block_1, block_2 in zip(tensor_map_1.blocks(), tensor_map_2.blocks()):
            values_1 = block_1.values
            values_2 = block_2.values
            loss += self.weight * self.losses["values"](values_1, values_2)
            for gradient_name, gradient_weight in self.gradient_weights.items():
                values_1 = block_1.gradient(gradient_name).values
                values_2 = block_2.gradient(gradient_name).values
                loss += gradient_weight * self.losses[gradient_name](values_1, values_2)

        return loss


class TensorMapDictLoss:
    """A loss function that operates on two ``Dict[str, metatensor.torch.TensorMap]``.

    At initialization, the user specifies a list of keys to use for the loss,
    along with a weight for each key.

    The loss is then computed as a weighted sum. Any keys that are not present
    in the dictionaries are ignored.

    :param weights: A dictionary mapping keys to weights. This might contain
        gradient keys, in the form ``<output_name>_<gradient_name>_gradients``.
    :param reduction: The reduction to apply to the loss.
        See :py:class:`torch.nn.MSELoss`.

    :returns: The loss as a zero-dimensional :py:class:`torch.Tensor`
        (with one entry).
    """

    def __init__(
        self,
        weights: Dict[str, float],
        reduction: str = "sum",
        type: Union[str, dict] = "mse",
    ):
        outputs = [key for key in weights.keys() if "gradients" not in key]
        self.losses = {}
        for output in outputs:
            value_weight = weights[output]
            gradient_weights = {}
            for key, weight in weights.items():
                if key.startswith(output) and key.endswith("_gradients"):
                    gradient_name = key.replace(f"{output}_", "").replace(
                        "_gradients", ""
                    )
                    gradient_weights[gradient_name] = weight
            type_output = _process_type(type, output)
            self.losses[output] = TensorMapLoss(
                reduction=reduction,
                weight=value_weight,
                gradient_weights=gradient_weights,
                type=type_output,
            )

    def __call__(
        self,
        tensor_map_dict_1: Dict[str, TensorMap],
        tensor_map_dict_2: Dict[str, TensorMap],
    ) -> torch.Tensor:
        # Assert that the two have the keys:
        assert set(tensor_map_dict_1.keys()) == set(tensor_map_dict_2.keys())

        # Initialize the loss:
        first_values = next(iter(tensor_map_dict_1.values())).block(0).values
        loss = torch.zeros((), dtype=first_values.dtype, device=first_values.device)

        # Compute the loss:
        for target in tensor_map_dict_1.keys():
            target_loss = self.losses[target](
                tensor_map_dict_1[target], tensor_map_dict_2[target]
            )
            loss += target_loss

        return loss


def _process_type(type: Union[str, DictConfig], output: str) -> Union[str, dict]:
    if not isinstance(type, str):
        assert "huber" in type
        # we process the Huber loss delta dict to make it similar to the
        # `weights` dict
        type_output = {"huber": {"deltas": {}}}  # type: ignore
        for key, delta in type["huber"]["deltas"].items():
            key_internal = to_internal_name(key)
            if key_internal == output:
                type_output["huber"]["deltas"]["values"] = delta
            elif key_internal.startswith(output) and key_internal.endswith(
                "_gradients"
            ):
                gradient_name = key_internal.replace(f"{output}_", "").replace(
                    "_gradients", ""
                )
                type_output["huber"]["deltas"][gradient_name] = delta
            else:
                pass
    else:
        type_output = type  # type: ignore
    return type_output
