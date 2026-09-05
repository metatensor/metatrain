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
from metatrain.utils.ensemble import uncertainty_output_name
from metatrain.utils.equivariance_penalty import equivariance_variance_output_name


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


class TensorMapEnsembleNLLLoss(LossInterface):
    r"""
    Gaussian negative log-likelihood loss for a shallow ensemble.

    Scores the ensemble *mean* prediction against the target, using the
    ensemble *variance* (spread across members) as the predictive variance:

    .. math::

        \mathrm{NLL} = \tfrac{1}{2}\left(
            \frac{(\mathrm{mean} - \mathrm{target})^2}{\mathrm{var}}
            + \log(\mathrm{var})
        \right)

    (delegates to :class:`torch.nn.GaussianNLLLoss`). Unlike a plain loss on
    the mean -- which only relies on independent random initialization to
    decorrelate members -- this actively rewards members whose spread tracks
    the actual error.

    Expects ``predictions`` to contain, alongside the usual mean prediction
    under ``name``, a ``{name}_uncertainty`` entry holding the ensemble
    variance with the *same* block/component/property shape (see
    :func:`metatrain.utils.ensemble.uncertainty_output_name` and
    :meth:`metatrain.pet.modules.backend.PETBackend.predict`) -- not a
    member-flattened tensor.

    :param name: key in the predictions/targets dict.
    :param gradient: must be ``None``; gradients are not supported.
    :param weight: weight of the loss contribution in the final aggregation.
    :param reduction: reduction mode for ``torch.nn.GaussianNLLLoss``.
    :param eps: numerical floor added to the variance before taking its log or
        dividing by it, for stability early in training.
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        reduction: str,
        eps: float = 1e-6,
    ):
        super().__init__(name, gradient, weight, reduction)
        if gradient is not None:
            raise ValueError(
                "'ensemble_nll' loss does not support gradient targets "
                f"(got gradient={gradient!r} for target {name!r})"
            )
        self.torch_loss = torch.nn.GaussianNLLLoss(reduction=reduction, eps=eps)

    @staticmethod
    def _flatten(tensor_map: TensorMap) -> torch.Tensor:
        return torch.cat([block.values.reshape(-1) for block in tensor_map.blocks()])

    def compute(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        :param predictions: must contain both ``self.target`` (the ensemble mean)
            and its ``_uncertainty`` counterpart (the ensemble variance).
        :param targets: mapping of names to :py:class:`TensorMap`.
        :param extra_data: ignored.
        :return: scalar torch.Tensor loss.
        """
        uncertainty_name = uncertainty_output_name(self.target)
        if uncertainty_name not in predictions:
            raise ValueError(
                f"'ensemble_nll' loss for {self.target!r} requires the "
                f"{uncertainty_name!r} prediction: enable 'shallow_ensemble' in "
                "the model hyperparameters"
            )

        mean_flat = self._flatten(predictions[self.target])
        target_flat = self._flatten(targets[self.target])
        var_flat = self._flatten(predictions[uncertainty_name])

        # Don't include in the loss calculation any points where the target is NaN
        not_nan = ~torch.isnan(target_flat)
        mean_flat = mean_flat[not_nan]
        target_flat = target_flat[not_nan]
        var_flat = var_flat[not_nan]

        if target_flat.numel() == 0:
            return torch.zeros((), dtype=mean_flat.dtype, device=mean_flat.device)

        return self.torch_loss(mean_flat, target_flat, var_flat)


class EquivariancePenaltyLoss(LossInterface):
    r"""
    On-line equivariance-error penalty: ``MSE(mean, target) + weight *
    mean(variance)``, where mean and variance are taken over
    ``num_augmentations`` random O(3) augmentations of each system.

    This trains directly against a model's own equivariance error, rather
    than only measuring it after the fact (as ``mtt eval``'s ``equivariance``
    option does, via ``metatomic.torch.o3.SymmetrizedModel``): each system is
    evaluated ``num_augmentations`` times under independent random rotations
    (and, unless the group is restricted, reflections), every prediction is
    mapped back to the system's original frame, and the resulting spread is
    penalized directly. Unlike ``SymmetrizedModel``'s exact (but expensive)
    quadrature integral, this is a cheap, unbiased *random-sample* estimate of
    the same quantity, suited to being computed every training step rather
    than only at evaluation time.

    This loss does not perform the augmentation itself -- a single
    ``LossInterface.compute()`` call only ever sees already-computed
    predictions, and producing them here needs ``num_augmentations`` separate
    forward passes. That is the trainer's job (see
    ``metatrain.pet.trainer``): it detects a target configured with this loss,
    augments and evaluates the model accordingly, and passes this loss its
    mean prediction under ``name`` (exactly as for any other loss) and the
    corresponding variance under
    :func:`~metatrain.utils.equivariance_penalty.equivariance_variance_output_name`
    (mirroring how :class:`TensorMapEnsembleNLLLoss` consumes a shallow
    ensemble's variance, computed and exposed by the *model* rather than the
    trainer).

    :param name: key in the predictions/targets dict.
    :param gradient: must be ``None``; gradients are not supported (mapping a
        gradient prediction back to the original frame after augmentation is
        not implemented).
    :param weight: weight of the loss contribution in the final aggregation.
    :param reduction: ``"mean"`` or ``"sum"``, applied to both terms.
    :param num_augmentations: number of independent random augmentations per
        system. Must be ``>= 2`` (a variance needs at least two samples).
    :param variance_weight: weight of the variance term relative to the MSE
        term.
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        reduction: str,
        num_augmentations: Optional[int] = None,
        variance_weight: Optional[float] = None,
    ):
        super().__init__(name, gradient, weight, reduction)
        if gradient is not None:
            raise NotImplementedError(
                "'equivariance_penalty' loss does not support gradient targets "
                f"(got gradient={gradient!r} for target {name!r})"
            )
        if reduction not in ("mean", "sum"):
            raise ValueError(
                "'equivariance_penalty' loss only supports reduction 'mean' or "
                f"'sum', got {reduction!r}"
            )
        if num_augmentations is None or num_augmentations < 2:
            raise ValueError(
                f"'equivariance_penalty' loss on target {name!r} requires "
                "'num_augmentations' >= 2 (got "
                f"{num_augmentations!r})"
            )
        if variance_weight is None:
            raise ValueError(
                f"'equivariance_penalty' loss on target {name!r} requires "
                "'variance_weight', the weight of the variance term relative to "
                "the MSE term"
            )
        self.num_augmentations = num_augmentations
        self.variance_weight = variance_weight
        self._mse = torch.nn.MSELoss(reduction=reduction)

    @staticmethod
    def _flatten(tensor_map: TensorMap) -> torch.Tensor:
        return torch.cat([block.values.reshape(-1) for block in tensor_map.blocks()])

    def compute_components(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the two terms this loss combines, separately, mainly so the
        trainer can log them individually (see ``metatrain.pet.trainer``): the
        plain MSE against the target, and the (unweighted, i.e. not yet
        multiplied by ``self.variance_weight``) mean/sum of the augmentation
        variance.

        :param predictions: must contain both ``self.target`` (the mean over
            augmentations) and its
            :func:`~metatrain.utils.equivariance_penalty
            .equivariance_variance_output_name` counterpart (the variance over
            augmentations).
        :param targets: mapping of names to :py:class:`TensorMap`.
        :return: ``(mse, variance_penalty)``.
        """
        variance_name = equivariance_variance_output_name(self.target)
        if variance_name not in predictions:
            raise RuntimeError(
                f"'equivariance_penalty' loss for {self.target!r} requires the "
                f"{variance_name!r} prediction; this is only produced by "
                "metatrain.pet.trainer's on-line augmentation machinery, which "
                "should already be active whenever this loss is configured -- "
                "this is an internal wiring bug if seen otherwise"
            )

        mean_flat = self._flatten(predictions[self.target])
        target_flat = self._flatten(targets[self.target])
        variance_flat = self._flatten(predictions[variance_name])

        # Don't include in the loss calculation any points where the target is NaN
        not_nan = ~torch.isnan(target_flat)
        mean_flat = mean_flat[not_nan]
        target_flat = target_flat[not_nan]
        variance_flat = variance_flat[not_nan]

        if target_flat.numel() == 0:
            zero = torch.zeros((), dtype=mean_flat.dtype, device=mean_flat.device)
            return zero, zero

        mse = self._mse(mean_flat, target_flat)
        variance_penalty = (
            variance_flat.mean() if self.reduction == "mean" else variance_flat.sum()
        )
        return mse, variance_penalty

    def compute(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        :param predictions: must contain both ``self.target`` (the mean over
            augmentations) and its
            :func:`~metatrain.utils.equivariance_penalty
            .equivariance_variance_output_name` counterpart (the variance over
            augmentations).
        :param targets: mapping of names to :py:class:`TensorMap`.
        :param extra_data: ignored.
        :return: scalar torch.Tensor loss.
        """
        mse, variance_penalty = self.compute_components(predictions, targets)
        return mse + self.variance_weight * variance_penalty


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
    POINTWISE = ("pointwise", BaseTensorMapLoss)
    MASKED_POINTWISE = ("masked_pointwise", MaskedTensorMapLoss)
    SHIFT_AGNOSTIC_MSE = ("shift_agnostic_mse", ShiftAgnosticMSE)
    GAUSSIAN_NLL = ("gaussian_nll_ensemble", TensorMapGaussianNLLLoss)
    GAUSSIAN_CRPS = ("gaussian_crps_ensemble", TensorMapGaussianCRPSLoss)
    EMPIRICAL_CRPS = ("empirical_crps_ensemble", TensorMapEmpiricalCRPSLoss)
    ENSEMBLE_NLL = ("ensemble_nll", TensorMapEnsembleNLLLoss)
    EQUIVARIANCE_PENALTY = ("equivariance_penalty", EquivariancePenaltyLoss)

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
