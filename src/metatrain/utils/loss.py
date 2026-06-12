# mypy: disable-error-code=misc
# We ignore misc errors in this file because TypedDict
# with default values is not allowed by mypy.
import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Literal, Optional, Type

import metatensor.torch as mts
import torch
import torch.nn.functional as F
from metatensor.torch import Labels, TensorBlock, TensorMap
from pydantic import ConfigDict, with_config
from torch.nn.modules.loss import _Loss
from typing_extensions import NotRequired, TypedDict

from metatrain.utils.data import TargetInfo


@with_config(ConfigDict(extra="allow"))
class LossParams(TypedDict):
    type: NotRequired[str] = "mse"
    weight: NotRequired[float] = 1.0
    reduction: NotRequired[Literal["none", "mean", "sum"]] = "mean"


@with_config(ConfigDict(extra="allow"))
class LossSpecification(TypedDict):
    type: NotRequired[str] = "mse"
    weight: NotRequired[float] = 1.0
    reduction: NotRequired[Literal["none", "mean", "sum"]] = "mean"
    gradients: NotRequired[dict[str, LossParams]] = {}


class LossInterface(ABC):
    """
    Abstract base for all loss functions.

    Subclasses must implement the ``compute`` method.

    :param name: key in the predictions/targets dict to select the TensorMap.
    :param gradient: optional name of a gradient field to extract.
    :param weight: weight of the loss contribution in the final aggregation.
    :param reduction: reduction mode for torch losses ("mean", "sum", etc.).
    """

    weight: float
    reduction: str
    loss_kwargs: Dict[str, Any]
    target: str
    gradient: Optional[str]

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        reduction: str,
    ) -> None:
        self.target = name
        self.gradient = gradient
        self.weight = weight
        self.reduction = reduction
        self.loss_kwargs = {}
        super().__init__()

    @abstractmethod
    def compute(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Compute the loss value.

        :param predictions: mapping from target names to the predictions
            for those targets.
        :param targets: mapping from target names to the reference targets.
        :param extra_data: Any extra data needed for the loss computation.

        :return: Value of the loss.
        """
        ...

    def __call__(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Alias to compute() for direct invocation.

        :param predictions: mapping from target names to the predictions
            for those targets.
        :param targets: mapping from target names to the reference targets.
        :param extra_data: Any extra data needed for the loss computation.

        :return: Value of the loss.
        """
        return self.compute(predictions, targets, extra_data)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "LossInterface":
        """
        Instantiate a loss from a config dict.

        :param cfg: keyword args matching the loss constructor.
        :return: instance of a LossInterface subclass.
        """
        return cls(**cfg)


# --- specific losses ------------------------------------------------------------------


class BaseTensorMapLoss(LossInterface):
    """
    Backbone for pointwise losses on :py:class:`TensorMap` entries.

    Provides a compute_flattened() helper that extracts values or gradients,
    flattens them, applies an optional mask, and computes the torch loss.

    :param name: key in the predictions/targets dict.
    :param gradient: optional gradient field name.
    :param weight: dummy here; real weighting in ScheduledLoss.
    :param reduction: reduction mode for torch loss.
    :param loss_fn: pre-instantiated torch.nn loss (e.g. MSELoss).
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        reduction: str,
        *,
        loss_fn: _Loss,
    ):
        super().__init__(name, gradient, weight, reduction)
        self.torch_loss = loss_fn

    def compute_flattened(
        self,
        tensor_map_predictions_for_target: TensorMap,
        tensor_map_targets_for_target: TensorMap,
        tensor_map_mask_for_target: Optional[TensorMap] = None,
    ) -> torch.Tensor:
        """
        Flatten prediction and target blocks (and optional mask), then
        apply the torch loss.

        :param tensor_map_predictions_for_target: predicted :py:class:`TensorMap`.
        :param tensor_map_targets_for_target: target :py:class:`TensorMap`.
        :param tensor_map_mask_for_target: optional mask :py:class:`TensorMap`.
        :return: scalar torch.Tensor of the computed loss.
        """
        list_of_prediction_segments = []
        list_of_target_segments = []

        def extract_flattened_values_from_block(
            tensor_block: mts.TensorBlock,
        ) -> torch.Tensor:
            """
            Extract values or gradients from a block, flatten to 1D.

            :param tensor_block: input :py:class:`TensorBlock`.
            :return: flattened torch.Tensor.
            """
            if self.gradient is not None:
                values = tensor_block.gradient(self.gradient).values
            else:
                values = tensor_block.values
            return values.reshape(-1)

        # Loop over each key in the TensorMap
        for single_key in tensor_map_predictions_for_target.keys:
            block_for_prediction = tensor_map_predictions_for_target.block(single_key)
            block_for_target = tensor_map_targets_for_target.block(single_key)

            flattened_prediction = extract_flattened_values_from_block(
                block_for_prediction
            )
            flattened_target = extract_flattened_values_from_block(block_for_target)

            if tensor_map_mask_for_target is not None:
                # Apply boolean mask if provided
                block_for_mask = tensor_map_mask_for_target.block(single_key)
                flattened_mask = extract_flattened_values_from_block(
                    block_for_mask
                ).bool()
                flattened_prediction = flattened_prediction[flattened_mask]
                flattened_target = flattened_target[flattened_mask]

            list_of_prediction_segments.append(flattened_prediction)
            list_of_target_segments.append(flattened_target)

        # Concatenate all segments and apply the torch loss
        all_predictions_flattened = torch.cat(list_of_prediction_segments)
        all_targets_flattened = torch.cat(list_of_target_segments)

        # Don't include in the loss calculation any points where
        # the target is NaN
        not_nan = ~torch.isnan(all_targets_flattened)
        all_targets_flattened = all_targets_flattened[not_nan]
        all_predictions_flattened = all_predictions_flattened[not_nan]

        if len(all_targets_flattened) == 0:
            # No valid data points to compute the loss
            return torch.zeros(
                (),
                dtype=all_predictions_flattened.dtype,
                device=all_predictions_flattened.device,
            )

        return self.torch_loss(all_predictions_flattened, all_targets_flattened)

    def compute(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Compute the unmasked pointwise loss.

        :param predictions: mapping of names to :py:class:`TensorMap`.
        :param targets: mapping of names to :py:class:`TensorMap`.
        :param extra_data: ignored for unmasked losses.
        :return: scalar torch.Tensor loss.
        """
        tensor_map_pred = predictions[self.target]
        tensor_map_targ = targets[self.target]

        # Check gradients are present in the target TensorMap
        if self.gradient is not None:
            if self.gradient not in tensor_map_targ[0].gradients_list():
                # Skip loss computation if block gradient is missing in the dataset
                # Tensor gradients are not tracked
                return torch.zeros(
                    (), dtype=torch.float, device=tensor_map_targ[0].values.device
                )
        return self.compute_flattened(tensor_map_pred, tensor_map_targ)


class MaskedTensorMapLoss(BaseTensorMapLoss):
    """
    Pointwise masked loss on :py:class:`TensorMap` entries.

    Inherits flattening and torch-loss logic from BaseTensorMapLoss.
    """

    def compute(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Dict[str, TensorMap]] = None,
    ) -> torch.Tensor:
        """
        Gather and flatten target and prediction blocks, then compute loss.

        :param predictions: Mapping from target names to TensorMaps.
        :param targets: Mapping from target names to TensorMaps.
        :param extra_data: Additional data for loss computation. Assumes that, for the
            target ``name`` used in the constructor, there is a corresponding data field
            ``name + "_mask"`` that contains the tensor to be used for masking. It
            should have the same metadata as the target and prediction tensors.
        :return: Scalar loss tensor.
        """
        mask_key = f"{self.target}_mask"
        if extra_data is None or mask_key not in extra_data:
            raise ValueError(
                f"Expected extra_data to contain TensorMap under '{mask_key}'"
            )
        tensor_map_pred = predictions[self.target]
        tensor_map_targ = targets[self.target]
        tensor_map_mask = extra_data[mask_key]
        return self.compute_flattened(tensor_map_pred, tensor_map_targ, tensor_map_mask)


# ------------------------------------------------------------------------
# Simple explicit subclasses for common pointwise losses
# ------------------------------------------------------------------------


class TensorMapMSELoss(BaseTensorMapLoss):
    """
    Unmasked mean-squared error on :py:class:`TensorMap` entries.

    :param name: key in the predictions/targets dict.
    :param gradient: optional gradient field name.
    :param weight: weight of the loss contribution in the final aggregation.
    :param reduction: reduction mode for torch loss.
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        reduction: str,
    ):
        super().__init__(
            name,
            gradient,
            weight,
            reduction,
            loss_fn=torch.nn.MSELoss(reduction=reduction),
        )


class TensorMapMAELoss(BaseTensorMapLoss):
    """
    Unmasked mean-absolute error on :py:class:`TensorMap` entries.

    :param name: key in the predictions/targets dict.
    :param gradient: optional gradient field name.
    :param weight: weight of the loss contribution in the final aggregation.
    :param reduction: reduction mode for torch loss.
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        reduction: str,
    ):
        super().__init__(
            name,
            gradient,
            weight,
            reduction,
            loss_fn=torch.nn.L1Loss(reduction=reduction),
        )


class TensorMapHuberLoss(BaseTensorMapLoss):
    """
    Unmasked Huber loss on :py:class:`TensorMap` entries.

    :param name: key in the predictions/targets dict.
    :param gradient: optional gradient field name.
    :param weight: weight of the loss contribution in the final aggregation.
    :param reduction: reduction mode for torch loss.
    :param delta: threshold parameter for HuberLoss.
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        reduction: str,
        delta: float,
    ):
        super().__init__(
            name,
            gradient,
            weight,
            reduction,
            loss_fn=torch.nn.HuberLoss(reduction=reduction, delta=delta),
        )


class TensorMapMaskedMSELoss(MaskedTensorMapLoss):
    """
    Masked mean-squared error on :py:class:`TensorMap` entries.

    :param name: key in the predictions/targets dict.
    :param gradient: optional gradient field name.
    :param weight: weight of the loss contribution in the final aggregation.
    :param reduction: reduction mode for torch loss.
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        reduction: str,
    ):
        super().__init__(
            name,
            gradient,
            weight,
            reduction,
            loss_fn=torch.nn.MSELoss(reduction=reduction),
        )


class TensorMapMaskedMAELoss(MaskedTensorMapLoss):
    """
    Masked mean-absolute error on :py:class:`TensorMap` entries.

    :param name: key in the predictions/targets dict.
    :param gradient: optional gradient field name.
    :param weight: weight of the loss contribution in the final aggregation.
    :param reduction: reduction mode for torch loss.
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        reduction: str,
    ):
        super().__init__(
            name,
            gradient,
            weight,
            reduction,
            loss_fn=torch.nn.L1Loss(reduction=reduction),
        )


class TensorMapMaskedHuberLoss(MaskedTensorMapLoss):
    """
    Masked Huber loss on :py:class:`TensorMap` entries.

    :param name: key in the predictions/targets dict.
    :param gradient: optional gradient field name.
    :param weight: weight of the loss contribution in the final aggregation.
    :param reduction: reduction mode for torch loss.
    :param delta: threshold parameter for HuberLoss.
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        reduction: str,
        delta: float,
    ):
        super().__init__(
            name,
            gradient,
            weight,
            reduction,
            loss_fn=torch.nn.HuberLoss(reduction=reduction, delta=delta),
        )


class WeightedTensorMapLoss(BaseTensorMapLoss):
    """
    Per-sample-component weighted pointwise loss on :py:class:`TensorMap` entries.

    This loss multiplies the per-element contribution of an underlying pointwise
    loss (MSE, MAE, Huber, ...) by a per-sample-component weight before reducing.
    For an ``mse`` base and ``reduction="mean"`` this computes

    .. math::

        L = \\frac{1}{N} \\sum_i w_i \\, (y_i - \\hat{y}_i)^2

    where :math:`w_i` is the weight of the :math:`i`-th sample-component, :math:`y_i`
    the reference value and :math:`\\hat{y}_i` the prediction.

    The weights are read from ``extra_data`` under the key ``f"{name}_weights"``. The
    weight :py:class:`TensorMap` must mirror the structure of the target (same blocks,
    components, properties and gradients), so that, once flattened, each prediction
    element has a matching weight. Weights are conventionally populated by setting
    ``sample_weight_key`` in the target (and gradient) sections of the training set
    configuration, which broadcasts a single per-sample weight over the components and
    properties of each block.

    .. note::

        The weights are stored as ``extra_data`` and are therefore **not** rescaled by
        the per-atom averaging applied to predictions and targets before the loss
        (e.g. the energy is divided by the number of atoms). The weight simply
        multiplies the already-(per-atom-)scaled squared error, so filling every weight
        with a constant :math:`c` reproduces the unweighted loss scaled by :math:`c`.

    :param name: key in the predictions/targets dict.
    :param gradient: optional gradient field name.
    :param weight: weight of the loss contribution in the final aggregation.
    :param reduction: reduction mode for the weighted loss ("mean", "sum" or "none").
    :param loss_fn: pre-instantiated torch.nn loss constructed with
        ``reduction="none"`` so the per-element values can be weighted before reduction.
    """

    weights_suffix: str = "_weights"

    def _reduce(self, values: torch.Tensor) -> torch.Tensor:
        """Apply the configured reduction to per-element (weighted) values.

        :param values: per-element (already weighted) loss values.
        :return: the reduced loss according to ``self.reduction``.
        """
        if self.reduction == "mean":
            return values.mean()
        elif self.reduction == "sum":
            return values.sum()
        elif self.reduction == "none":
            return values
        else:
            raise ValueError(f"{self.reduction!r} is not a valid reduction")

    def compute_weighted(
        self,
        tensor_map_predictions_for_target: TensorMap,
        tensor_map_targets_for_target: TensorMap,
        tensor_map_weights_for_target: TensorMap,
    ) -> torch.Tensor:
        """
        Flatten prediction, target and weight blocks, apply the per-element torch
        loss, weight it, and reduce.

        :param tensor_map_predictions_for_target: predicted :py:class:`TensorMap`.
        :param tensor_map_targets_for_target: target :py:class:`TensorMap`.
        :param tensor_map_weights_for_target: weight :py:class:`TensorMap`, mirroring
            the structure of the target.
        :return: scalar (or per-element, if ``reduction="none"``) torch.Tensor.
        """
        list_of_prediction_segments = []
        list_of_target_segments = []
        list_of_weight_segments = []

        def extract_flattened_values_from_block(
            tensor_block: mts.TensorBlock,
        ) -> torch.Tensor:
            if self.gradient is not None:
                values = tensor_block.gradient(self.gradient).values
            else:
                values = tensor_block.values
            return values.reshape(-1)

        for single_key in tensor_map_predictions_for_target.keys:
            block_for_prediction = tensor_map_predictions_for_target.block(single_key)
            block_for_target = tensor_map_targets_for_target.block(single_key)
            block_for_weight = tensor_map_weights_for_target.block(single_key)

            flattened_prediction = extract_flattened_values_from_block(
                block_for_prediction
            )
            flattened_target = extract_flattened_values_from_block(block_for_target)

            # The weight block mirrors the target structure. If the (gradient) weight
            # is missing we fall back to ones, i.e. an unweighted contribution.
            if self.gradient is not None and self.gradient not in (
                block_for_weight.gradients_list()
            ):
                flattened_weight = torch.ones_like(flattened_target)
            else:
                flattened_weight = extract_flattened_values_from_block(block_for_weight)

            list_of_prediction_segments.append(flattened_prediction)
            list_of_target_segments.append(flattened_target)
            list_of_weight_segments.append(flattened_weight)

        all_predictions_flattened = torch.cat(list_of_prediction_segments)
        all_targets_flattened = torch.cat(list_of_target_segments)
        all_weights_flattened = torch.cat(list_of_weight_segments)

        # Don't include in the loss calculation any points where the target is NaN
        not_nan = ~torch.isnan(all_targets_flattened)
        all_targets_flattened = all_targets_flattened[not_nan]
        all_predictions_flattened = all_predictions_flattened[not_nan]
        all_weights_flattened = all_weights_flattened[not_nan]

        if len(all_targets_flattened) == 0:
            # No valid data points to compute the loss
            return torch.zeros(
                (),
                dtype=all_predictions_flattened.dtype,
                device=all_predictions_flattened.device,
            )

        # self.torch_loss uses reduction="none", so this is a per-element tensor
        per_element_loss = self.torch_loss(
            all_predictions_flattened, all_targets_flattened
        )
        weighted_loss = all_weights_flattened * per_element_loss
        return self._reduce(weighted_loss)

    def compute(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Dict[str, TensorMap]] = None,
    ) -> torch.Tensor:
        """
        Gather prediction, target and weight blocks, then compute the weighted loss.

        :param predictions: Mapping from target names to TensorMaps.
        :param targets: Mapping from target names to TensorMaps.
        :param extra_data: Additional data for the loss computation. Must contain a
            weight :py:class:`TensorMap` under the key ``f"{name}_weights"`` mirroring
            the structure of the target. These weights are conventionally provided by
            setting ``sample_weight_key`` in the training set configuration.
        :return: Scalar loss tensor.
        """
        weights_key = f"{self.target}{self.weights_suffix}"
        if extra_data is None or weights_key not in extra_data:
            raise ValueError(
                f"Expected extra_data to contain a weight TensorMap under "
                f"'{weights_key}'. Set 'sample_weight_key' in the target (and, if "
                f"needed, gradient) configuration of the training set to use a "
                f"weighted loss such as '{self.target}'."
            )
        tensor_map_targ = targets[self.target]

        # Check gradients are present in the target TensorMap
        if self.gradient is not None:
            if self.gradient not in tensor_map_targ[0].gradients_list():
                # Skip loss computation if block gradient is missing in the dataset
                return torch.zeros(
                    (), dtype=torch.float, device=tensor_map_targ[0].values.device
                )

        return self.compute_weighted(
            predictions[self.target],
            tensor_map_targ,
            extra_data[weights_key],
        )


class TensorMapWeightedMSELoss(WeightedTensorMapLoss):
    """
    Per-sample-component weighted mean-squared error on :py:class:`TensorMap` entries.

    See :py:class:`WeightedTensorMapLoss` for details. The per-sample weights are read
    from ``extra_data`` under the key ``f"{name}_weights"``.

    :param name: key in the predictions/targets dict.
    :param gradient: optional gradient field name.
    :param weight: weight of the loss contribution in the final aggregation.
    :param reduction: reduction mode for the weighted loss.
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        reduction: str,
    ):
        super().__init__(
            name,
            gradient,
            weight,
            reduction,
            loss_fn=torch.nn.MSELoss(reduction="none"),
        )


class TensorMapWeightedMAELoss(WeightedTensorMapLoss):
    """
    Per-sample-component weighted mean-absolute error on :py:class:`TensorMap` entries.

    See :py:class:`WeightedTensorMapLoss` for details.

    :param name: key in the predictions/targets dict.
    :param gradient: optional gradient field name.
    :param weight: weight of the loss contribution in the final aggregation.
    :param reduction: reduction mode for the weighted loss.
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        reduction: str,
    ):
        super().__init__(
            name,
            gradient,
            weight,
            reduction,
            loss_fn=torch.nn.L1Loss(reduction="none"),
        )


class TensorMapWeightedHuberLoss(WeightedTensorMapLoss):
    """
    Per-sample-component weighted Huber loss on :py:class:`TensorMap` entries.

    See :py:class:`WeightedTensorMapLoss` for details.

    :param name: key in the predictions/targets dict.
    :param gradient: optional gradient field name.
    :param weight: weight of the loss contribution in the final aggregation.
    :param reduction: reduction mode for the weighted loss.
    :param delta: threshold parameter for HuberLoss.
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        reduction: str,
        delta: float,
    ):
        super().__init__(
            name,
            gradient,
            weight,
            reduction,
            loss_fn=torch.nn.HuberLoss(reduction="none", delta=delta),
        )


class ShiftAgnosticMSE(LossInterface):
    """
    Shift agnostic MSE loss on :py:class:`TensorMap` entries.

    This loss assumes that the target is some kind of profile
    along the properties of the ``TensorBlock``. It finds the
    rigid shift between the predictions and targets that
    minimizes the MSE, and returns that minimal MSE.

    :param name: key for the target in the prediction/target dictionary.
    :param gradient: optional gradient field name.
    :param weight: weight of the loss contribution in the final aggregation.
    :param int_weight: The loss function can also contain the MSE on the
      cumulative profile. This number weights the contribution of the
      cumulative term in the final loss. If 0, no cumulative term is added.
    :param grad_penalty_weight: The loss function penalizes gradients of the
      predicted profiles in the regions where the target is NaN.
      This number weights the contribution of the penalty term
      in the final loss. If 0, the predictions on those regions are
      free to be what they want.
    :param reduction: reduction mode for torch loss.
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        int_weight: float,
        grad_penalty_weight: float,
        reduction: str,
    ):
        super().__init__(
            name,
            gradient,
            weight,
            reduction,
        )
        self.grad_penalty_weight = grad_penalty_weight
        self.int_weight = int_weight

        interval = 0.05
        self.grid = (
            (torch.tensor([1 / 4, -4 / 3, 3.0, -4.0, 25 / 12]) / interval)
            .unsqueeze(dim=(0))
            .unsqueeze(dim=(0))
            .float()
        )

    def compute(
        self,
        model_predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Any | None = None,
    ) -> torch.Tensor:
        """
        Gather and flatten target and prediction blocks, then compute shift
        agnostic loss.

        :param model_predictions: Mapping from target names to TensorMaps.
        :param targets: Mapping from target names to TensorMaps.
        :param extra_data: extra data, not needed for this loss function

        :return: Scalar loss tensor.
        """

        tensor_map_pred = model_predictions[self.target]
        tensor_map_targ = targets[self.target]

        # There should only be one block

        predictions = tensor_map_pred.block().values.float()
        convolution_pad = torch.zeros_like(predictions)
        predictions = torch.hstack([convolution_pad, predictions, convolution_pad])

        target = tensor_map_targ.block().values.float()
        mask = (~torch.isnan(target)).float()
        target = torch.nan_to_num(target)

        dtype = predictions.dtype
        device = predictions.device
        # Uses convolutions to find the optimal shift that minimzes the MSE
        # between the prediction and the target
        sum_sq_smaller = torch.sum((target**2) * mask, dim=1, keepdim=True)
        batch_size = predictions.shape[0]
        bigger_reshaped = predictions.unsqueeze(0)
        kernel = (target * mask).unsqueeze(1)
        cross_corr = F.conv1d(bigger_reshaped, kernel, groups=batch_size)
        cross_corr = cross_corr.squeeze(0)
        bigger_sq_reshaped = (predictions**2).unsqueeze(0)
        mask_kernel = mask.unsqueeze(1)
        sum_sq_bigger = F.conv1d(bigger_sq_reshaped, mask_kernel, groups=batch_size)
        sum_sq_bigger = sum_sq_bigger.squeeze(0)
        losses = sum_sq_bigger - 2 * cross_corr + sum_sq_smaller
        losses = torch.clamp(losses, min=0.0)
        front_tail = torch.cumsum(predictions**2, dim=1)
        shape_difference = predictions.shape[1] - target.shape[1]
        additional_error = torch.hstack(
            [
                torch.zeros(len(predictions), device=predictions.device).reshape(-1, 1),
                front_tail[:, :shape_difference],
            ]
        )
        total_losses = losses + additional_error
        final_loss, shift = torch.min(total_losses, dim=1)

        loss = torch.mean(final_loss)
        # Compute gradient loss
        aligned_predictions = []
        adjusted_mask = []
        for index, prediction in enumerate(predictions):
            aligned_prediction = prediction[
                shift[index] : shift[index] + len(target[0])
            ]
            mask_i = torch.hstack(  # Adjust the mask to account for the discrete shift
                [
                    (torch.ones(shift[index])).bool().to(device),
                    mask[index].bool().to(device),
                    torch.zeros(
                        int(predictions.shape[1] - len(mask[index]) - shift[index])
                    )
                    .bool()
                    .to(device),
                ]
            )
            aligned_predictions.append(aligned_prediction)
            adjusted_mask.append(mask_i)
        aligned_predictions = torch.vstack(aligned_predictions)
        adjusted_mask = torch.vstack(adjusted_mask)
        if self.grad_penalty_weight > 0:
            grad_predictions = torch.nn.functional.conv1d(
                predictions.unsqueeze(dim=1), self.grid.to(device).to(dtype)
            ).squeeze(dim=1)

            dim_loss = (
                predictions.shape[1] - grad_predictions.shape[1]
            )  # Dimensions lost due to the gradient convolution
            gradient_loss = (
                torch.mean(
                    torch.trapezoid(
                        (
                            (grad_predictions * (~adjusted_mask[:, dim_loss:])) ** 2
                        ),  # non-zero gradients outside the window are penalized
                        dx=0.05,
                        dim=1,
                    )
                )
                * self.grad_penalty_weight
            )
        else:
            gradient_loss = 0.0
        if self.int_weight > 0:
            int_predictions = torch.cumulative_trapezoid(
                aligned_predictions, dx=0.05, dim=1
            )
            int_target = torch.cumulative_trapezoid(target, dx=0.05, dim=1)
            int_error = (int_predictions - int_target) ** 2
            int_error = int_error * mask[:, 1:].unsqueeze(
                dim=1
            )  # only penalize the integral where the target is defined
            int_MSE = (
                torch.mean(torch.trapezoid(int_error, dx=0.05, dim=1)) * self.int_weight
            )
        else:
            int_MSE = 0.0

        return loss + gradient_loss + int_MSE


class TensorMapEnsembleLoss(BaseTensorMapLoss):
    """
    Loss for ensembles based on :py:class:`TensorMap` entries.
    Assumes that ensemble is the outermost dimension of :py:class:`TensorBlock`
    properties.

    :param name: key in the predictions/targets dict.
    :param gradient: optional gradient field name.
    :param weight: weight of the loss contribution in the final aggregation.
    :param reduction: reduction mode for torch loss.
    :param loss_fn: pre-instantiated torch.nn loss.
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        reduction: str,
        loss_fn: torch.nn.Module,
    ):
        super().__init__(
            name,
            gradient,
            weight,
            reduction,
            loss_fn=loss_fn,
        )

    # this is technically incompatible with the BaseTensorMapLoss compute_flattened:
    # ignore the type error
    def compute_flattened(  # type: ignore[override]
        self,
        pred_mean: TensorMap,
        target: TensorMap,
        pred_var: TensorMap,
    ) -> torch.Tensor:
        """
        Flatten prediction and target blocks (and optional mask), then
        apply the torch loss.

        :param pred_mean: mean of ensemble predictions :py:class:`TensorMap`.
        :param target: target :py:class:`TensorMap`.
        :param pred_var: variance of ensemble predictions :py:class:`TensorMap`.
        :return: scalar torch.Tensor of the computed loss.
        """
        if self.gradient is not None:
            return 0.0  # gradients not supported for this loss yet

        list_pred_mean_segments = []
        list_target_segments = []
        list_pred_var_segments = []

        def extract_flattened_values_from_block(
            tensor_block: mts.TensorBlock,
        ) -> torch.Tensor:
            """
            Extract values or gradients from a block, flatten to 1D.

            :param tensor_block: input :py:class:`TensorBlock`.
            :return: flattened torch.Tensor.
            """
            values = tensor_block.values
            return values.reshape(-1)

        # Loop over each key in the TensorMap
        for single_key in target.keys:
            block_pred_mean = pred_mean.block(single_key)
            block_target = target.block(single_key)
            block_pred_var = pred_var.block(single_key)

            flat_pred_mean = extract_flattened_values_from_block(block_pred_mean)
            flat_target = extract_flattened_values_from_block(block_target)
            flat_pred_var = extract_flattened_values_from_block(block_pred_var)

            list_pred_mean_segments.append(flat_pred_mean)
            list_target_segments.append(flat_target)
            list_pred_var_segments.append(flat_pred_var)

        # Concatenate all segments and apply the torch loss
        all_pred_mean_flattened = torch.cat(list_pred_mean_segments)
        all_targets_flattened = torch.cat(list_target_segments)
        all_pred_var_flattened = torch.cat(list_pred_var_segments)

        return self.torch_loss(
            all_pred_mean_flattened,
            all_targets_flattened,
            all_pred_var_flattened,
        )

    def compute(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Dict[str, TensorMap]] = None,
    ) -> torch.Tensor:
        """
        Gather and flatten target and prediction blocks, then compute loss.

        :param predictions: Mapping from target names to TensorMaps, must contain
            ensemble as the outer-most property dimension.
        :param targets: Mapping from target names to their ref value TensorMaps.
        :param extra_data: Ignored for this loss.
        :return: Scalar loss tensor.
        """

        ens_name = "mtt::aux::" + self.target.replace("mtt::", "") + "_ensemble"
        if ens_name == "mtt::aux::energy_ensemble":
            ens_name = "energy_ensemble"

        tmap_pred_orig = predictions[self.target]
        tmap_pred_ens = predictions[ens_name]
        tmap_targ = targets[self.target]

        # number of ensembles extracted from TensorMaps
        n_ens = (
            tmap_pred_ens.block(0).values.shape[1]
            // tmap_pred_orig.block(0).values.shape[1]
        )

        ens_pred_values = tmap_pred_ens.block().values  # shape: samples, properties

        ens_pred_values = ens_pred_values.reshape(ens_pred_values.shape[0], n_ens, -1)
        ens_pred_mean = ens_pred_values.mean(dim=1)
        ens_pred_var = ens_pred_values.var(dim=1, unbiased=True)

        tmap_pred_mean = TensorMap(
            keys=Labels(
                names=["_"],
                values=torch.tensor([[0]], device=tmap_targ.block().values.device),
            ),
            blocks=[
                TensorBlock(
                    values=ens_pred_mean,
                    samples=tmap_targ.block().samples,
                    components=tmap_targ.block().components,
                    properties=tmap_targ.block().properties,
                ),
            ],
        )

        tmap_pred_var = TensorMap(
            keys=Labels(
                names=["_"],
                values=torch.tensor([[0]], device=tmap_targ.block().values.device),
            ),
            blocks=[
                TensorBlock(
                    values=ens_pred_var,
                    samples=tmap_targ.block().samples,
                    components=tmap_targ.block().components,
                    properties=tmap_targ.block().properties,
                ),
            ],
        )

        # Note that we're ignoring all gradients for now. This can be extended later.
        return self.compute_flattened(tmap_pred_mean, tmap_targ, tmap_pred_var)


class GaussianCRPSLoss(torch.nn.Module):
    r"""
    Gaussian CRPS loss.

    This implements the closed-form expression for the CRPS of a Gaussian predictive
    distribution :math:`\mathcal{N}(\mu, \sigma^2)` evaluated at a target value
    :math:`x`:

    .. math::

        \text{CRPS}(x; \mu, \sigma) =
        \sigma \left[ z(2\Phi(z) - 1) + 2\phi(z) - \frac{1}{\sqrt{\pi}} \right]

    where :math:`z = \frac{x - \mu}{\sigma}`, :math:`\Phi` is the standard normal CDF,
    and :math:`\phi` is the standard normal PDF.

    :param reduction: 'none', 'mean', or 'sum'.
    :param eps: small constant for numerical stability on variance.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-12):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Gaussian CRPS loss.

        :param input: Mean predictions.
        :param target: Target values.
        :param var: Variance of the predictions.
        :return: Value of the loss.
        """

        var_clamped = torch.clamp(var, min=self.eps)
        sigma = torch.sqrt(var_clamped)

        # z = (x - mu) / sigma
        z = (target - input) / sigma

        # standard normal pdf and cdf
        # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
        # phi(z) = 1/sqrt(2*pi) * exp(-z^2 / 2)
        sqrt_2 = math.sqrt(2.0)
        inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
        inv_sqrt_pi = 1.0 / math.sqrt(math.pi)

        phi = inv_sqrt_2pi * torch.exp(-0.5 * z**2)
        Phi = 0.5 * (1.0 + torch.erf(z / sqrt_2))

        crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - inv_sqrt_pi)

        if self.reduction == "mean":
            return crps.mean()
        elif self.reduction == "sum":
            return crps.sum()
        elif self.reduction == "none":
            return crps
        else:
            raise ValueError(self.reduction + " is not valid")


class EmpiricalCRPSLoss(torch.nn.Module):
    r"""
    Empirical CRPS loss for ensemble predictions.

    The ensemble predictions :math:`\{Y_i\}_{i=1}^M` for each data point define
    an empirical predictive distribution:

    .. math::

        F_M(y) = \frac{1}{M} \sum_{i=1}^M \mathbb{1}_{Y_i \le y}

    The CRPS of this empirical distribution at observation :math:`z` has the
    closed form:

    .. math::

        \text{CRPS}(F_M, z) =
        \frac{1}{M} \sum_{i=1}^M |Y_i - z| - \frac{1}{2 M^2} \sum_{i,j} |Y_i - Y_j|

    :param reduction: 'none', 'mean', or 'sum'.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        ensemble: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Empirical CRPS loss.

        :param ensemble: Ensemble predictions, shape (B, M).
        :param target: Target values, shape (B,).
        :return: Value of the loss.
        """
        if ensemble.dim() != 2:
            raise ValueError(
                f"EmpiricalCRPSLoss expects ensemble with shape (B, M), "
                f"got {ensemble.shape}"
            )
        if target.dim() != 1 or target.shape[0] != ensemble.shape[0]:
            raise ValueError(
                f"EmpiricalCRPSLoss expects target with shape (B,), "
                f"got {target.shape} for ensemble batch {ensemble.shape[0]}"
            )

        # mean |Y_i - z| over ensemble members
        term1 = (ensemble - target.unsqueeze(1)).abs().mean(dim=1)

        # 0.5 * mean |Y_i - Y_j| over all pairs (i, j)
        diffs = ensemble.unsqueeze(2) - ensemble.unsqueeze(1)
        term2 = 0.5 * diffs.abs().mean(dim=(1, 2))

        crps = term1 - term2

        if self.reduction == "mean":
            return crps.mean()
        elif self.reduction == "sum":
            return crps.sum()
        elif self.reduction == "none":
            return crps
        else:
            raise ValueError(self.reduction + " is not valid")


class TensorMapGaussianNLLLoss(TensorMapEnsembleLoss):
    """
    Gaussian negative log-likelihood loss for :py:class:`TensorMap` entries.

    :param name: key in the predictions/targets dict.
    :param gradient: optional gradient field name.
    :param weight: weight of the loss contribution in the final aggregation.
    :param reduction: reduction mode for torch loss.
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        reduction: str,
    ):
        super().__init__(
            name,
            gradient,
            weight,
            reduction,
            loss_fn=torch.nn.GaussianNLLLoss(reduction=reduction),
        )


class TensorMapGaussianCRPSLoss(TensorMapEnsembleLoss):
    """
    Gaussian CRPS loss for :py:class:`TensorMap` entries.

    :param name: key in the predictions/targets dict.
    :param gradient: optional gradient field name.
    :param weight: weight of the loss contribution in the final aggregation.
    :param reduction: reduction mode for torch loss.
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        reduction: str,
    ):
        super().__init__(
            name,
            gradient,
            weight,
            reduction,
            loss_fn=GaussianCRPSLoss(reduction=reduction),
        )


class TensorMapEmpiricalCRPSLoss(TensorMapEnsembleLoss):
    """
    Empirical CRPS loss for :py:class:`TensorMap` entries.

    :param name: key in the predictions/targets dict.
    :param gradient: optional gradient field name.
    :param weight: weight of the loss contribution in the final aggregation.
    :param reduction: reduction mode for torch loss.
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        reduction: str,
    ):
        super().__init__(
            name,
            gradient,
            weight,
            reduction,
            loss_fn=EmpiricalCRPSLoss(reduction=reduction),
        )

    # we need to override compute to handle empirical CRPS
    def compute(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Dict[str, TensorMap]] = None,
    ) -> torch.Tensor:
        """
        Gather and flatten target and prediction blocks, then compute loss.

        :param predictions: Mapping from target names to TensorMaps, must contain
            ensemble as the outer-most property dimension.
        :param targets: Mapping from target names to their ref value TensorMaps.
        :param extra_data: Ignored for this loss.
        :return: Scalar loss tensor.
        """

        ens_name = "mtt::aux::" + self.target.replace("mtt::", "") + "_ensemble"
        if ens_name == "mtt::aux::energy_ensemble":
            ens_name = "energy_ensemble"

        tmap_pred_orig = predictions[self.target]
        tmap_pred_ens = predictions[ens_name]
        tmap_targ = targets[self.target]

        # number of ensembles extracted from TensorMaps
        n_ens = (
            tmap_pred_ens.block(0).values.shape[1]
            // tmap_pred_orig.block(0).values.shape[1]
        )

        ens_pred_values = tmap_pred_ens.block().values  # shape: samples, properties
        ens_pred_values = ens_pred_values.reshape(ens_pred_values.shape[0], n_ens, -1)

        # For empirical CRPS, we need the full ensemble predictions
        target_values = tmap_targ.block().values  # (S, P)

        S, M, P = ens_pred_values.shape

        # Reorder to (S, P, M) and then flatten S*P into B:
        # y_ensemble: (B, M), y_target: (B,)
        y_ensemble = ens_pred_values.permute(0, 2, 1).reshape(-1, M)
        y_target = target_values.reshape(-1)

        return self.torch_loss(y_ensemble, y_target)


# --- aggregator -----------------------------------------------------------------------


class LossAggregator(LossInterface):
    """
    Aggregate multiple :py:class:`LossInterface` terms with scheduled weights and
    metadata.

    :param targets: mapping from target names to :py:class:`TargetInfo`.
    :param config: per-target configuration dict.
    """

    def __init__(
        self, targets: Dict[str, TargetInfo], config: Dict[str, LossSpecification]
    ):
        super().__init__(name="", gradient=None, weight=0.0, reduction="mean")
        self.losses: Dict[str, LossInterface] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

        for target_name, target_info in targets.items():
            target_config = config.get(
                target_name,
                LossSpecification(
                    {
                        "type": "mse",
                        "weight": 1.0,
                        "reduction": "mean",
                        "gradients": {},
                    }
                ),
            )

            # Create main loss and its scheduler
            base_loss = create_loss(
                target_config["type"],
                name=target_name,
                gradient=None,
                weight=target_config["weight"],
                reduction=target_config["reduction"],
                **{
                    pname: pval
                    for pname, pval in target_config.items()
                    if pname
                    not in (
                        "type",
                        "weight",
                        "reduction",
                        "gradients",
                    )
                },
            )
            self.losses[target_name] = base_loss
            self.metadata[target_name] = {
                "type": target_config["type"],
                "weight": base_loss.weight,
                "reduction": base_loss.reduction,
                "gradients": {},
            }
            for pname, pval in target_config.items():
                if pname not in (
                    "type",
                    "weight",
                    "reduction",
                    "gradients",
                ):
                    self.metadata[target_name][pname] = pval

            # Create gradient-based losses
            gradient_config = target_config["gradients"]
            for gradient_name in target_info.layout[0].gradients_list():
                gradient_key = f"{target_name}_grad_{gradient_name}"

                gradient_specific_config = gradient_config.get(
                    gradient_name,
                    LossSpecification(
                        {
                            "type": "mse",
                            "weight": 1.0,
                            "reduction": "mean",
                        }
                    ),
                )

                grad_loss = create_loss(
                    gradient_specific_config["type"],
                    name=target_name,
                    gradient=gradient_name,
                    weight=gradient_specific_config["weight"],
                    reduction=gradient_specific_config["reduction"],
                    **{
                        pname: pval
                        for pname, pval in gradient_specific_config.items()
                        if pname
                        not in (
                            "type",
                            "weight",
                            "reduction",
                            "gradients",
                        )
                    },
                )
                self.losses[gradient_key] = grad_loss
                self.metadata[target_name]["gradients"][gradient_name] = {
                    "type": gradient_specific_config["type"],
                    "weight": grad_loss.weight,
                    "reduction": grad_loss.reduction,
                }
                for pname, pval in gradient_specific_config.items():
                    if pname not in (
                        "type",
                        "weight",
                        "reduction",
                        "gradients",
                    ):
                        self.metadata[target_name]["gradients"][gradient_name][
                            pname
                        ] = pval

    def compute(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Sum over all scheduled losses present in the predictions.

        :param predictions: mapping from target names to :py:class:`TensorMap`.
        :param targets: mapping from target names to :py:class:`TensorMap`.
        :param extra_data: Any extra data needed for the loss computation.
        :return: scalar torch.Tensor with the total loss.
        """
        # Initialize a zero tensor matching the dtype and device of the first block
        first_tensor_map = next(iter(predictions.values()))
        first_block = first_tensor_map.block(first_tensor_map.keys[0])
        total_loss = torch.zeros(
            (), dtype=first_block.values.dtype, device=first_block.values.device
        )

        # Sum each scheduled term that has a matching prediction
        for term in self.losses.values():
            if term.target not in predictions:
                continue
            total_loss = total_loss + term.weight * term.compute(
                predictions, targets, extra_data
            )

        return total_loss


class LossType(Enum):
    """
    Enumeration of available loss types and their implementing classes.

    :param key: string key for the loss type.
    :param cls: class implementing the loss type.
    """

    MSE = ("mse", TensorMapMSELoss)
    MAE = ("mae", TensorMapMAELoss)
    HUBER = ("huber", TensorMapHuberLoss)
    MASKED_MSE = ("masked_mse", TensorMapMaskedMSELoss)
    MASKED_MAE = ("masked_mae", TensorMapMaskedMAELoss)
    MASKED_HUBER = ("masked_huber", TensorMapMaskedHuberLoss)
    WEIGHTED_MSE = ("weighted_mse", TensorMapWeightedMSELoss)
    WEIGHTED_MAE = ("weighted_mae", TensorMapWeightedMAELoss)
    WEIGHTED_HUBER = ("weighted_huber", TensorMapWeightedHuberLoss)
    POINTWISE = ("pointwise", BaseTensorMapLoss)
    MASKED_POINTWISE = ("masked_pointwise", MaskedTensorMapLoss)
    SHIFT_AGNOSTIC_MSE = ("shift_agnostic_mse", ShiftAgnosticMSE)
    GAUSSIAN_NLL = ("gaussian_nll_ensemble", TensorMapGaussianNLLLoss)
    GAUSSIAN_CRPS = ("gaussian_crps_ensemble", TensorMapGaussianCRPSLoss)
    EMPIRICAL_CRPS = ("empirical_crps_ensemble", TensorMapEmpiricalCRPSLoss)

    def __init__(self, key: str, cls: Type[LossInterface]) -> None:
        self._key = key
        self._cls = cls

    @property
    def key(self) -> str:
        """String key for this loss type."""
        return self._key

    @property
    def cls(self) -> Type[LossInterface]:
        """Class implementing this loss type."""
        return self._cls

    @classmethod
    def from_key(cls, key: str) -> "LossType":
        """
        Look up a LossType by its string key.

        :param key: key that identifies the loss type.
        :raises ValueError: if the key is not valid.
        :return: the matching LossType enum member.
        """
        for loss_type in cls:
            if loss_type.key == key:
                return loss_type
        valid_keys = ", ".join(loss_type.key for loss_type in cls)
        raise ValueError(f"Unknown loss '{key}'. Valid types: {valid_keys}")


def create_loss(
    loss_type: str,
    *,
    name: str,
    gradient: Optional[str],
    weight: float,
    reduction: str,
    **extra_kwargs: Any,
) -> LossInterface:
    r"""
    Factory to instantiate a concrete :py:class:`LossInterface` given its string key.

    :param loss_type: string key matching one of the members of :py:class:`LossType`.
    :param name: target name for the loss.
    :param gradient: gradient name, if present.
    :param weight: weight of the loss contribution in the final aggregation.
    :param reduction: reduction mode for the torch loss.
    :param \*\*extra_kwargs: additional hyperparameters specific to the loss type.
    :return: instance of the selected loss.
    """
    loss_type_entry = LossType.from_key(loss_type)
    try:
        return loss_type_entry.cls(
            name=name,
            gradient=gradient,
            weight=weight,
            reduction=reduction,
            **extra_kwargs,
        )
    except TypeError as e:
        raise TypeError(f"Error constructing loss '{loss_type}': {e}") from e
