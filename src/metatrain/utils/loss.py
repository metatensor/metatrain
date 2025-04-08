from typing import Dict, Optional, Tuple, Union

import torch
from metatensor.torch import TensorMap
from omegaconf import DictConfig
from torch.nn.modules.loss import _Loss

from metatrain.utils.external_naming import to_internal_name


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
    :param sliding_factor: The factor to apply to the exponential moving average
        of the "sliding" weights. These are weights that act on different components of
        the loss (for example, energies and forces), based on their individual recent
        history. If ``None``, no sliding weights are used in the computation of the
        loss.
    :param type: The type of loss to use. This can be either "mse" or "mae".
        A Huber loss can also be requested as a dictionary with the key "huber" and
        the value must be a dictionary with the key "deltas" and the value
        must be a dictionary with the keys "values" and the gradient keys.
        The values of the dictionary must be the deltas to use for the
        Huber loss.

    :returns: The loss as a zero-dimensional :py:class:`torch.Tensor`
        (with one entry).
    """

    def __init__(
        self,
        reduction: str = "mean",
        weight: float = 1.0,
        gradient_weights: Optional[Dict[str, float]] = None,
        sliding_factor: Optional[float] = None,
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
        self.sliding_factor = sliding_factor
        self.sliding_weights: Optional[Dict[str, TensorMap]] = None

    def __call__(
        self,
        predictions_tensor_map: TensorMap,
        targets_tensor_map: TensorMap,
    ) -> Tuple[torch.Tensor, Dict[str, Tuple[float, int]]]:
        # Check that the two have the same metadata, except for the samples,
        # which can be different due to batching, but must have the same size:
        if predictions_tensor_map.keys != targets_tensor_map.keys:
            raise ValueError(
                "TensorMapSlidingLoss requires the two TensorMaps to have the "
                "same keys."
            )
        for block_1, block_2 in zip(
            predictions_tensor_map.blocks(), targets_tensor_map.blocks()
        ):
            if block_1.properties != block_2.properties:
                raise ValueError(
                    "TensorMapSlidingLoss requires the two TensorMaps to have the same "
                    "properties."
                )
            if block_1.components != block_2.components:
                raise ValueError(
                    "TensorMapSlidingLoss requires the two TensorMaps to have the same "
                    "components."
                )
            if len(block_1.samples) != len(block_2.samples):
                raise ValueError(
                    "TensorMapSlidingLoss requires the two TensorMaps "
                    "to have the same number of samples."
                )
            for gradient_name in self.gradient_weights.keys():
                if len(block_1.gradient(gradient_name).samples) != len(
                    block_2.gradient(gradient_name).samples
                ):
                    raise ValueError(
                        "TensorMapSlidingLoss requires the two TensorMaps "
                        "to have the same number of gradient samples."
                    )
                if (
                    block_1.gradient(gradient_name).properties
                    != block_2.gradient(gradient_name).properties
                ):
                    raise ValueError(
                        "TensorMapSlidingLoss requires the two TensorMaps "
                        "to have the same gradient properties."
                    )
                if (
                    block_1.gradient(gradient_name).components
                    != block_2.gradient(gradient_name).components
                ):
                    raise ValueError(
                        "TensorMapSlidingLoss requires the two TensorMaps "
                        "to have the same gradient components."
                    )

        # First time the function is called: compute the sliding weights only
        # from the targets (if they are enabled)
        if self.sliding_factor is not None and self.sliding_weights is None:
            self.sliding_weights = get_sliding_weights(
                self.losses,
                self.sliding_factor,
                targets_tensor_map,
            )

        # Compute the loss:
        loss = torch.zeros(
            (),
            dtype=predictions_tensor_map.block(0).values.dtype,
            device=predictions_tensor_map.block(0).values.device,
        )
        for key in targets_tensor_map.keys:
            block_1 = predictions_tensor_map.block(key)
            block_2 = targets_tensor_map.block(key)
            values_1 = block_1.values
            values_2 = block_2.values
            # sliding weights: default to 1.0 if not used/provided for this target
            sliding_weight = (
                1.0
                if self.sliding_weights is None
                else self.sliding_weights.get("values", 1.0)
            )
            loss += (
                self.weight * self.losses["values"](values_1, values_2) / sliding_weight
            )
            for gradient_name, gradient_weight in self.gradient_weights.items():
                values_1 = block_1.gradient(gradient_name).values
                values_2 = block_2.gradient(gradient_name).values
                # sliding weights: default to 1.0 if not used/provided for this target
                sliding_weigths_value = (
                    1.0
                    if self.sliding_weights is None
                    else self.sliding_weights.get(gradient_name, 1.0)
                )
                loss += (
                    gradient_weight
                    * self.losses[gradient_name](values_1, values_2)
                    / sliding_weigths_value
                )
        if self.sliding_factor is not None:
            self.sliding_weights = get_sliding_weights(
                self.losses,
                self.sliding_factor,
                targets_tensor_map,
                predictions_tensor_map,
                self.sliding_weights,
            )
        return loss


class TensorMapDictLoss:
    """A loss function that operates on two ``Dict[str, metatensor.torch.TensorMap]``.

    At initialization, the user specifies a list of keys to use for the loss,
    along with a weight for each key.

    The loss is then computed as a weighted sum. Any keys that are not present
    in the dictionaries are ignored.

    :param weights: A dictionary mapping keys to weights. This might contain
        gradient keys, in the form ``<output_name>_<gradient_name>_gradients``.
    :param sliding_factor: The factor to apply to the exponential moving average
        of the "sliding" weights. These are weights that act on different components of
        the loss (for example, energies and forces), based on their individual recent
        history. If ``None``, no sliding weights are used in the computation of the
        loss.
    :param reduction: The reduction to apply to the loss.
        See :py:class:`torch.nn.MSELoss`.

    :returns: The loss as a zero-dimensional :py:class:`torch.Tensor`
        (with one entry).
    """

    def __init__(
        self,
        weights: Dict[str, float],
        sliding_factor: Optional[float] = None,
        reduction: str = "mean",
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
            if output == "energy" and sliding_factor is not None:
                self.losses[output] = TensorMapLoss(
                    reduction=reduction,
                    weight=value_weight,
                    gradient_weights=gradient_weights,
                    sliding_factor=sliding_factor,
                    type=type_output,
                )
            else:
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


def get_sliding_weights(
    losses: Dict[str, _Loss],
    sliding_factor: float,
    targets: TensorMap,
    predictions: Optional[TensorMap] = None,
    previous_sliding_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute the sliding weights for the loss function.

    The sliding weights are computed as the absolute difference between the
    predictions and the targets.

    :param predictions: The predictions.
    :param targets: The targets.

    :return: The sliding weights.
    """
    sliding_weights = {}
    if predictions is None:
        for block in targets.blocks():
            values = block.values
            sliding_weights["values"] = (
                losses["values"](values, values.mean() * torch.ones_like(values)) + 1e-6
            )
            for gradient_name, gradient_block in block.gradients():
                values = gradient_block.values
                sliding_weights[gradient_name] = losses[gradient_name](
                    values, torch.zeros_like(values)
                )
    elif predictions is not None:
        if previous_sliding_weights is None:
            raise RuntimeError(
                "previous_sliding_weights must be provided if predictions is not None"
            )
        else:
            for predictions_block, target_block in zip(
                predictions.blocks(), targets.blocks()
            ):
                target_values = target_block.values
                predictions_values = predictions_block.values
                sliding_weights["values"] = (
                    sliding_factor * previous_sliding_weights["values"]
                    + (1 - sliding_factor)
                    * losses["values"](predictions_values, target_values).detach()
                )
                for gradient_name, gradient_block in target_block.gradients():
                    target_values = gradient_block.values
                    predictions_values = predictions_block.gradient(
                        gradient_name
                    ).values
                    sliding_weights[gradient_name] = (
                        sliding_factor * previous_sliding_weights[gradient_name]
                        + (1 - sliding_factor)
                        * losses[gradient_name](
                            predictions_values, target_values
                        ).detach()
                    )
    return sliding_weights


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
