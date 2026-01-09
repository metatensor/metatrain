from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Type

import metatensor.torch as mts
import torch
from metatensor.torch import TensorMap, Labels
from torch.nn.modules.loss import _Loss
import numpy as np
from metatrain.utils.data import TargetInfo
from metatrain.utils.output_gradient import compute_gradient
import scipy
from scipy.ndimage import gaussian_filter1d

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

        # For stress targets, we allow users to use NaN entries for systems without
        # PBCs or with mixed PBCs. We filter them out here.
        if self.gradient == "strain" or "non_conservative_stress" in self.target:
            valid_mask = ~torch.isnan(all_targets_flattened)
            all_predictions_flattened = all_predictions_flattened[valid_mask]
            all_targets_flattened = all_targets_flattened[valid_mask]

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


class MaskedDOSLoss(LossInterface):
    """
    Masked DOS loss on :py:class:`TensorMap` entries.

    :param name: key for the dos in the prediction/target dictionary.
    :param gradient: optional gradient field name.
    :param weight: weight of the loss contribution in the final aggregation.
    :param grad_weight: Multiplier for the gradient of the unmasked DOS component.
    :param int_weight: Multiplier for the cumulative DOS component.
    :param extra_targets: Number of extra targets predicted by the model.
    :param reduction: reduction mode for torch loss.
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        grad_weight: float,
        int_weight: float,
        extra_targets: int,
        reduction: str,
    ):
        super().__init__(
            name,
            gradient,
            weight,
            reduction,
        )
        self.grad_weight = grad_weight
        self.int_weight = int_weight
        self.extra_targets = extra_targets

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
        extra_data: Optional[Dict[str, TensorMap]] = None,
    ) -> torch.Tensor:
        """
        Gather and flatten target and prediction blocks, then compute loss.

        :param model_predictions: Mapping from target names to TensorMaps.
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

        tensor_map_pred = model_predictions[self.target]
        tensor_map_targ = targets[self.target]
        tensor_map_mask = extra_data[mask_key]

        # There should only be one block

        predictions = tensor_map_pred.block().values
        target = tensor_map_targ.block().values[:, self.extra_targets :]
        mask = tensor_map_mask.block().values[:, self.extra_targets :].bool()

        device = predictions.device
        predictions_unfolded = predictions.unfold(1, len(target[0]), 1)
        target_expanded = target[:, None, :]
        delta = target_expanded - predictions_unfolded
        dynamic_delta = delta * mask.unsqueeze(dim=1)
        losses = torch.trapezoid(dynamic_delta * dynamic_delta, dx=0.05, dim=2)
        front_tail = torch.cumulative_trapezoid(predictions**2, dx=0.05, dim=1)
        additional_error = torch.hstack(
            [
                torch.zeros(len(predictions), device=device).reshape(-1, 1),
                front_tail[:, : self.extra_targets],
            ]
        )
        total_losses = losses + additional_error
        final_loss, shift = torch.min(total_losses, dim=1)
        dos_loss = torch.mean(final_loss)
        # Compute gradient loss
        aligned_predictions = []
        adjusted_dos_mask = []
        for index, prediction in enumerate(predictions):
            aligned_prediction = prediction[
                shift[index] : shift[index] + len(target[0])
            ]
            dos_mask_i = (
                torch.hstack(  # Adjust the mask to account for the discrete shift
                    [
                        (torch.ones(shift[index])).bool().to(device),
                        mask[index],
                        (torch.zeros(int(self.extra_targets - shift[index])))
                        .bool()
                        .to(device),
                    ]
                )
            )
            aligned_predictions.append(aligned_prediction)
            adjusted_dos_mask.append(dos_mask_i)
        aligned_predictions = torch.vstack(aligned_predictions)
        adjusted_dos_mask = torch.vstack(adjusted_dos_mask)
        if self.grad_weight > 0:
            grad_predictions = torch.nn.functional.conv1d(
                predictions.unsqueeze(dim=1), self.grid.to(device)
            ).squeeze(dim=1)
            dim_loss = (
                predictions.shape[1] - grad_predictions.shape[1]
            )  # Dimensions lost due to the gradient convolution
            gradient_loss = (
                torch.mean(
                    torch.trapezoid(
                        (
                            (grad_predictions * (~adjusted_dos_mask[:, dim_loss:])) ** 2
                        ),  # non-zero gradients outside the window are penalized
                        dx=0.05,
                        dim=1,
                    )
                )
                * self.grad_weight
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
            )  # only penalize the integral where the DOS is defined
            int_MSE = (
                torch.mean(torch.trapezoid(int_error, dx=0.05, dim=1)) * self.int_weight
            )
        else:
            int_MSE = 0.0

        return dos_loss + gradient_loss + int_MSE


class BandgapLoss(LossInterface):
    """
    Masked DOS loss on :py:class:`TensorMap` entries.

    :param name: key for the dos in the prediction/target dictionary.
    :param gradient: optional gradient field name.
    :param weight: weight of the loss contribution in the final aggregation.
    :param force_weight: Multiplier for the forces of the gap.
    :param force: Whether to apply the force term.
    :param reduction: reduction mode for torch loss.
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        force_weight: float,
        force: bool,
        reduction: str,
    ):
        super().__init__(
            name,
            gradient,
            weight,
            reduction,
        )
        self.force_weight = force_weight
        self.force = force
        self.print = True

    def compute(
        self,
        model_predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        """
        gap_key = f"mtt::gap"
        gapforce_key = f"mtt::gapforce"
        systems, model, extra_data = extra_data
        if extra_data is None or gap_key not in extra_data:
            raise ValueError(
                f"Expected extra_data to contain TensorMap under '{gap_key}'"
            )
        if self.force:
            if gapforce_key not in extra_data:
                raise ValueError(
                    f"Expected extra_data to contain TensorMap under '{gapforce_key}'"
                )
        
        tensor_map_pred = model_predictions[self.target] # should be the gapdos prediction
        tensor_map_gap = extra_data[gap_key]
        if self.force:
            tensor_map_gapforce = extra_data[gapforce_key]
            true_gapforce = tensor_map_gapforce.block().values

        # There should only be one block
        gapdos_predictions = tensor_map_pred.block().values
        true_gap = tensor_map_gap.block().values

        bandgap_predictions = model.bandgap_layer(gapdos_predictions)
        if self.force:
             gap_force_predictions = torch.vstack(compute_gradient(bandgap_predictions, [system.positions for system in systems], is_training=True, destroy_graph= False))
        if self.print:
            print ("Printing Shapes")
            print ("Gapdos predictions:", gapdos_predictions.shape)
            print("Bandgap predictions:", bandgap_predictions.shape)
            print("Bandgap targets:", true_gap.shape)
            if self.force:
                print("Gap force predictions:", gap_force_predictions.shape)
                print("Gap force targets:", true_gapforce.shape)
            self.print = False
        
        total_loss = torch.mean((bandgap_predictions - true_gap) ** 2)
        if self.force:
            gapforce_loss = torch.mean((gap_force_predictions - true_gapforce) ** 2) 
            total_loss += gapforce_loss * self.force_weight
        return total_loss
    

class HOMOLUMOLoss(LossInterface):
    """
    HOMO LUMO loss on :py:class:`TensorMap` entries to get bandgap

    :param name: key for the dos in the prediction/target dictionary.
    :param gradient: optional gradient field name.
    :param weight: weight of the loss contribution in the final aggregation.
    :param force_weight: Multiplier for the forces of the gap.
    :param force: Whether to apply the force term.
    :param reduction: reduction mode for torch loss.
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        force_weight: float,
        force: bool,
        reduction: str,
    ):
        super().__init__(
            name,
            gradient,
            weight,
            reduction,
        )
        self.force_weight = force_weight
        self.force = force
        self.print = True

    def smoothmax(self, tensor, alpha=1.0, dim = 0):
        """
        Computes the Log-Sum-Exp smoothmax.
        alpha > 0: approximates max
        alpha < 0: approximates min
        """
        return torch.logsumexp(alpha * tensor, dim = dim) / alpha
        
    def compute(
        self,
        model_predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Dict[str, TensorMap]] = None,
    ) -> torch.Tensor:
        """
        Gather and flatten target and prediction blocks, then compute loss.

        :param model_predictions: Mapping from target names to TensorMaps.
        :param targets: Mapping from target names to TensorMaps.
        :param extra_data: Additional data for loss computation. Assumes that, for the
            target ``name`` used in the constructor, there is a corresponding data field
            ``name + "_mask"`` that contains the tensor to be used for masking. It
            should have the same metadata as the target and prediction tensors.
        :return: Scalar loss tensor.
        """
        gap_key = f"mtt::gap"
        gapforce_key = f"mtt::gapforce"
        systems, model, extra_data = extra_data
        if extra_data is None or gap_key not in extra_data:
            raise ValueError(
                f"Expected extra_data to contain TensorMap under '{gap_key}'"
            )
        if self.force:
            if gapforce_key not in extra_data:
                raise ValueError(
                    f"Expected extra_data to contain TensorMap under '{gapforce_key}'"
                )
            
        tensor_map_gap = extra_data[gap_key]
        if self.force:
            tensor_map_gapforce = extra_data[gapforce_key]
            true_gapforce = tensor_map_gapforce.block().values
        device = model_predictions['mtt::HOMO'].block().values.device
        # Use logsumexp to get smoothmax
        gaps = []
        for i in range(len(systems)):
            structural_HOMO_i = mts.slice(model_predictions['mtt::HOMO'], axis = 'samples', selection = Labels(names=['system'], values = torch.tensor([[i]]).to(device)))
            structural_LUMO_i = mts.slice(model_predictions['mtt::LUMO'], axis = 'samples', selection = Labels(names=['system'], values = torch.tensor([[i]]).to(device)))
            max_HOMO = self.smoothmax(structural_HOMO_i.block().values, alpha = 20).squeeze()
            min_LUMO = self.smoothmax(structural_LUMO_i.block().values, alpha = -20).squeeze()
            pred_gap_i = min_LUMO - max_HOMO
            gaps.append(pred_gap_i)
        gaps = torch.vstack(gaps)
        true_gaps = tensor_map_gap.block().values
        gap_MSE = torch.mean((gaps - true_gaps)**2)
        if self.force:
            gap_force_predictions = torch.vstack(compute_gradient(gaps, [system.positions for system in systems], is_training=True, destroy_graph= False))
            gap_force_MSE = torch.mean((gap_force_predictions - true_gapforce.squeeze())**2)
            gap_MSE += self.force_weight * gap_force_MSE

        return gap_MSE


class NoLoss(LossInterface):
    """
    No loss on :py:class:`TensorMap` entries to support fake targets.

    :param name: key for the dos in the prediction/target dictionary.
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
        )
    def compute(
        self,
        model_predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Dict[str, TensorMap]] = None,
    ) -> torch.Tensor:
        """
        returns zero loss
        """
        return torch.tensor(0.0)

class GapDOSLoss(LossInterface):
    """
    Masked GapDOS loss on :py:class:`TensorMap` entries.

    :param name: key for the dos in the prediction/target dictionary.
    :param gradient: optional gradient field name.
    :param weight: weight of the loss contribution in the final aggregation.
    :param force_weight: Multiplier for the forces of the gap.
    :param force: Whether to apply the force term.
    :param reduction: reduction mode for torch loss.
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        int_weight: float,
        gap_weight: float,
        extra_targets: int,
        reduction: str,
    ):
        super().__init__(
            name,
            gradient,
            weight,
            reduction,
        )
        self.gap_weight = gap_weight
        self.int_weight = int_weight
        self.extra_targets = extra_targets # should be 300 for this exercise
        self.print = True

    def compute(
        self,
        model_predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        """
        gap_key = f"mtt::gap"
        CDOS_key = f"mtt::cdos"
        mask_key = f"mtt::mask"
        systems, model, extra_data = extra_data
        if extra_data is None or gap_key not in extra_data:
            raise ValueError(
                f"Expected extra_data to contain TensorMap under '{gap_key}'"
            )
        if CDOS_key not in extra_data:
            raise ValueError(
                f"Expected extra_data to contain TensorMap under '{CDOS_key}'"
            )
        if mask_key not in extra_data:
            raise ValueError(
                f"Expected extra_data to contain TensorMap under '{mask_key}'"
            )
        
        tensor_map_pred = model_predictions[self.target] # should be the dos prediction (Normalized already)
        tensor_map_gap = extra_data[gap_key]
        tensor_map_CDOS = extra_data[CDOS_key]
        tensor_map_mask = extra_data[mask_key]
        tensor_map_targ = targets[self.target]
        # LAST DOS VALUE IS START OF CDOS
        # There should only be one block
        dos_predictions = tensor_map_pred.block().values[:,:-1]
        cdos_start_prediction = tensor_map_pred.block().values[:,-1]
        true_gap = tensor_map_gap.block().values # Already Normalized wrt atoms
        true_dos = tensor_map_targ.block().values[:, self.extra_targets+1:] # Already normalized wrt atoms
        true_CDOS = tensor_map_CDOS.block().values[:, self.extra_targets:] # Already Normalized wrt atoms
        mask = tensor_map_mask.block().values.bool()[:, self.extra_targets:]

        # Evaluate error on DOS
        device = dos_predictions.device
        predictions_unfolded = dos_predictions.unfold(1, len(true_dos[0]), 1)
        target_expanded = true_dos[:, None, :]
        delta = target_expanded - predictions_unfolded
        dynamic_delta = delta * mask.unsqueeze(dim=1)
        losses = torch.trapezoid(dynamic_delta * dynamic_delta, dx=0.05, dim=2)
        front_tail = torch.cumulative_trapezoid(dos_predictions**2, dx=0.05, dim=1)
        additional_error = torch.hstack(
            [
                torch.zeros(len(dos_predictions), device=device).reshape(-1, 1),
                front_tail[:, : self.extra_targets],
            ]
        )
        total_losses = losses + additional_error
        final_loss, shift = torch.min(total_losses, dim=1)
        dos_loss = torch.mean(final_loss)

        # CDOS loss
        pred_CDOS = torch.cumulative_trapezoid(dos_predictions, dx=0.05, dim=1) + cdos_start_prediction.unsqueeze(dim=1) # shape 700
        CDOS_MSE = 0
        for index_i, shift_i in enumerate(shift):
            pred_CDOS_i = pred_CDOS[index_i][shift_i + 1 : shift_i + 1 + len(true_CDOS[0])]
            CDOS_loss_i = torch.trapezoid((pred_CDOS_i - true_CDOS[index_i])**2, dx =0.05)
            CDOS_MSE += CDOS_loss_i

        CDOS_MSE = CDOS_MSE / len(systems)
        total_dos_loss = dos_loss + CDOS_MSE * self.int_weight

        # Bandgap loss
        bandgap_predictions = model.bandgap_layer(dos_predictions)
        # if self.force:
        #      gap_force_predictions = torch.vstack(compute_gradient(bandgap_predictions, [system.positions for system in systems], is_training=True, destroy_graph= False))
        if self.print:
            print ("Printing Shapes")
            print ("Mask shape:", mask.shape)
            print ("DOS predictions shape:", dos_predictions.shape)
            print ("True DOS shape:", true_dos.shape)
            print ("CDOS predictions shape:", pred_CDOS.shape)
            print ("true_CDOS shape:", true_CDOS.shape)
            print("Bandgap predictions:", bandgap_predictions.shape)
            print("Bandgap targets:", true_gap.shape)
            # if self.force:
            #     print("Gap force predictions:", gap_force_predictions.shape)
            #     print("Gap force targets:", true_gapforce.shape)
            self.print = False
        
        gap_loss = torch.mean((bandgap_predictions - true_gap) ** 2)
        # if self.force:
        #     gapforce_loss = torch.mean((gap_force_predictions - true_gapforce) ** 2) 
        #     gap_loss += gapforce_loss * self.force_weight
        total_loss = total_dos_loss + gap_loss * self.gap_weight
        return total_loss

class FocusedDOSLoss(LossInterface):
    """
    Focused DOS loss on :py:class:`TensorMap` entries.

    :param name: key for the dos in the prediction/target dictionary.
    :param gradient: optional gradient field name.
    :param weight: weight of the loss contribution in the final aggregation.
    :param force_weight: Multiplier for the forces of the gap.
    :param force: Whether to apply the force term.
    :param reduction: reduction mode for torch loss.
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        int_weight: float,
        extra_targets: int,
        reduction: str,
    ):
        super().__init__(
            name,
            gradient,
            weight,
            reduction,
        )
        self.int_weight = int_weight
        self.extra_targets = extra_targets # should be 300 for this exercise
        self.print = True

    def compute(
        self,
        model_predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        """
        gap_key = f"mtt::gap"
        CDOS_key = f"mtt::cdos"
        mask_key = f"mtt::mask"
        systems, model, extra_data = extra_data
        if CDOS_key not in extra_data:
            raise ValueError(
                f"Expected extra_data to contain TensorMap under '{CDOS_key}'"
            )
        if mask_key not in extra_data:
            raise ValueError(
                f"Expected extra_data to contain TensorMap under '{mask_key}'"
            )
        
        tensor_map_pred = model_predictions[self.target] # should be the dos prediction (Normalized already)
        tensor_map_CDOS = extra_data[CDOS_key]
        tensor_map_mask = extra_data[mask_key]
        tensor_map_targ = targets[self.target]
        # LAST DOS VALUE IS START OF CDOS
        # There should only be one block
        dos_predictions = tensor_map_pred.block().values[:,:-1]
        cdos_start_prediction = tensor_map_pred.block().values[:,-1]
        true_dos = tensor_map_targ.block().values[:, self.extra_targets+1:] # Already normalized wrt atoms
        true_CDOS = tensor_map_CDOS.block().values[:, self.extra_targets:] # Already Normalized wrt atoms
        mask = tensor_map_mask.block().values.bool()[:, self.extra_targets:]

        # Evaluate error on DOS
        device = dos_predictions.device
        predictions_unfolded = dos_predictions.unfold(1, len(true_dos[0]), 1)
        target_expanded = true_dos[:, None, :]
        delta = target_expanded - predictions_unfolded
        dynamic_delta = delta * mask.unsqueeze(dim=1)
        losses = torch.trapezoid(dynamic_delta * dynamic_delta, dx=0.05, dim=2)
        front_tail = torch.cumulative_trapezoid(dos_predictions**2, dx=0.05, dim=1)
        additional_error = torch.hstack(
            [
                torch.zeros(len(dos_predictions), device=device).reshape(-1, 1),
                front_tail[:, : self.extra_targets],
            ]
        )
        total_losses = losses + additional_error
        final_loss, shift = torch.min(total_losses, dim=1)
        dos_loss = torch.mean(final_loss)

        # CDOS loss
        pred_CDOS = torch.cumulative_trapezoid(dos_predictions, dx=0.05, dim=1) + cdos_start_prediction.unsqueeze(dim=1) # shape 700
        CDOS_MSE = 0
        for index_i, shift_i in enumerate(shift):
            pred_CDOS_i = pred_CDOS[index_i][shift_i + 1: shift_i + 1 + len(true_CDOS[0])]
            if len(pred_CDOS_i) < len(true_CDOS[0]):
                # Pad pred_CDOS_i with last value
                pad_size = len(true_CDOS[0]) - len(pred_CDOS_i)
                pad_values = pred_CDOS_i[-1].repeat(pad_size)
                pred_CDOS_i = torch.cat([pred_CDOS_i, pad_values], dim=0)
            CDOS_loss_i = torch.trapezoid((pred_CDOS_i - true_CDOS[index_i])**2, dx =0.05)
            CDOS_MSE += CDOS_loss_i

        CDOS_MSE = CDOS_MSE / len(systems)
        total_dos_loss = dos_loss + CDOS_MSE * self.int_weight

        # Bandgap loss
        # if self.force:
        #      gap_force_predictions = torch.vstack(compute_gradient(bandgap_predictions, [system.positions for system in systems], is_training=True, destroy_graph= False))
        if self.print:
            print ("Printing Shapes")
            print ("Mask shape:", mask.shape)
            print ("DOS predictions shape:", dos_predictions.shape)
            print ("True DOS shape:", true_dos.shape)
            print ("CDOS predictions shape:", pred_CDOS.shape)
            print ("true_CDOS shape:", true_CDOS.shape)
        # if self.force:
            #     print("Gap force predictions:", gap_force_predictions.shape)
            #     print("Gap force targets:", true_gapforce.shape)
            self.print = False
        
        # if self.force:
        #     gapforce_loss = torch.mean((gap_force_predictions - true_gapforce) ** 2) 
        #     gap_loss += gapforce_loss * self.force_weight
        total_loss = total_dos_loss
        return total_loss


class DifferentiableDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Calculate kernel parameters
        # Sigma from your code: 0.5 / 0.05 = 10.0
        sigma = 10.0
        
        # SciPy's gaussian_filter1d defaults to truncating at 4.0 sigmas
        truncate = 4.0
        radius = int(truncate * sigma + 0.5)
        
        # Generate the Gaussian kernel
        x_coord = torch.arange(-radius, radius + 1).float()
        kernel = torch.exp(-0.5 * (x_coord / sigma) ** 2)
        kernel = kernel / kernel.sum()  # Normalize
        
        # Reshape for conv1d: (Out_channels, In_channels, Kernel_size)
        # We register it as a buffer so it is saved with the model but not trained
        self.register_buffer('kernel', kernel.view(1, 1, -1))
        self.pad_size = radius

    def denoise(self, x):
        """
        x: Input tensor (Length,) or (Batch, Length)
        """
        # 1. Reshape input to (Batch, Channel, Length) for conv1d
        is_1d = x.dim() == 1
        if is_1d:
            x_input = x.view(1, 1, -1)
        elif x.dim() == 2:
            x_input = x.unsqueeze(1)
        else:
            x_input = x

        # 2. Pad using 'replicate' (equivalent to nearest-neighbor padding)
        x_padded = torch.nn.functional.pad(
            x_input, 
            (self.pad_size, self.pad_size), 
            mode='replicate'
        )
        device = x.device

        # 3. Apply Gaussian filter (differentiable)
        a = torch.nn.functional.conv1d(x_padded, self.kernel.to(device))

        # 4. Restore original shape
        if is_1d:
            a = a.squeeze()
        elif x.dim() == 2:
            a = a.squeeze(1)

        # 5. Apply Sigmoid and multiply
        # Removed torch.tensor() wrapper to preserve gradients
        return x * torch.sigmoid((a - 0.1) * 100)

def get_gap_differentiable(doses: torch.Tensor, cdos_starts: torch.Tensor, n_elecs: torch.Tensor, tol=0.1, dx=0.05, temp=100.0):
    """
    Calculates band gaps in a fully vectorized, differentiable way.
    'temp' controls the sharpness of the gap detection mask.
    Lower temp (e.g. 100) = Smoother gradients.
    Higher temp (e.g. 10000) = Sharper, binary-like cutoffs.
    """
    cdoses = torch.cumsum(doses, dim=1) * dx + cdos_starts.view(-1,1)
    
    # We detach this index calculation because discrete indexing is not differentiable.
    # Gradients flow through the gap *width*, not the *position* of Ef.
    Ef_indices = torch.argmax((cdoses > n_elecs.unsqueeze(1)).float(), dim=1).detach()

    # Soft mask: Sigmoid. 
    # If dose < tol -> val is ~1.0 (Gap). If dose > tol -> val is ~0.0 (No Gap).
    is_gap = torch.sigmoid(temp * (tol - doses))

    batch_size, n_grid = doses.shape
    device = doses.device
    
    # Create grid indices
    grid_indices = torch.arange(n_grid, device=device).unsqueeze(0).expand(batch_size, -1)
    Ef_indices_expanded = Ef_indices.unsqueeze(1)

    # Positional Masks
    mask_right = grid_indices >= Ef_indices_expanded # Include Ef in right sweep
    mask_left = grid_indices < Ef_indices_expanded   # Exclude Ef in left sweep

    # --- Right Side Sweep ---
    # Set everything to the LEFT of Ef to 1.0 so they don't block the cumprod flow
    right_input = is_gap.clone()
    right_input[~mask_right] = 1.0 
    
    # Cumprod flows Left -> Right. It stops multiplying (becomes 0) once it hits a high DOS region.
    right_connectivity = torch.cumprod(right_input, dim=1)
    right_gap_width = torch.sum(right_connectivity * mask_right, dim=1) * dx

    # --- Left Side Sweep ---
    # Set everything to the RIGHT of Ef to 1.0
    left_input = is_gap.clone()
    left_input[~mask_left] = 1.0
    
    # Cumprod flows Right -> Left (using flip)
    left_connectivity = torch.cumprod(left_input.flip(1), dim=1).flip(1)
    left_gap_width = torch.sum(left_connectivity * mask_left, dim=1) * dx

    return left_gap_width + right_gap_width

class PhysicalGapDOSLoss(LossInterface):
    """
    Focused DOS loss on :py:class:`TensorMap` entries.

    :param name: key for the dos in the prediction/target dictionary.
    :param gradient: optional gradient field name.
    :param weight: weight of the loss contribution in the final aggregation.
    :param force_weight: Multiplier for the forces of the gap.
    :param force: Whether to apply the force term.
    :param reduction: reduction mode for torch loss.
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        int_weight: float,
        gap_weight: float,
        extra_targets: int,
        reduction: str,
    ):
        super().__init__(
            name,
            gradient,
            weight,
            reduction,
        )
        self.int_weight = int_weight
        self.gap_weight = gap_weight
        self.extra_targets = extra_targets # should be 300 for this exercise
        self.sigmoid = torch.nn.Sigmoid()

        self.print = True
        self.denoiser = DifferentiableDenoiser()

        
    def compute(
        self,
        model_predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        """
        gap_key = f"mtt::gap"
        CDOS_key = f"mtt::cdos"
        mask_key = f"mtt::mask"
        elec_key = f"mtt::nelec"
        systems, model, extra_data = extra_data
        if extra_data is None or gap_key not in extra_data:
            raise ValueError(
                f"Expected extra_data to contain TensorMap under '{gap_key}'"
            )
        if CDOS_key not in extra_data:
            raise ValueError(
                f"Expected extra_data to contain TensorMap under '{CDOS_key}'"
            )
        if mask_key not in extra_data:
            raise ValueError(
                f"Expected extra_data to contain TensorMap under '{mask_key}'"
            )
        if elec_key not in extra_data:
            raise ValueError(
                f"Expected extra_data to contain TensorMap under '{elec_key}'"
            )
        
        tensor_map_pred = model_predictions[self.target] # should be the dos prediction (Normalized already)
        tensor_map_gap = extra_data[gap_key]
        tensor_map_CDOS = extra_data[CDOS_key]
        tensor_map_mask = extra_data[mask_key]
        tensor_map_elec = extra_data[elec_key]
        tensor_map_targ = targets[self.target]
        # LAST DOS VALUE IS START OF CDOS
        # There should only be one block
        dos_predictions = self.denoiser.denoise(tensor_map_pred.block().values[:,:-1])
        cdos_start_prediction = tensor_map_pred.block().values[:,-1]
        true_gap = tensor_map_gap.block().values # Already Normalized wrt atoms
        true_dos = tensor_map_targ.block().values[:, self.extra_targets+1:] # Already normalized wrt atoms
        true_CDOS = tensor_map_CDOS.block().values[:, self.extra_targets:] # Already Normalized wrt atoms
        mask = tensor_map_mask.block().values.bool()[:, self.extra_targets:]
        true_n_electrons = tensor_map_elec.block().values.squeeze()

        # Evaluate error on DOS
        device = dos_predictions.device
        predictions_unfolded = dos_predictions.unfold(1, len(true_dos[0]), 1)
        target_expanded = true_dos[:, None, :]
        delta = target_expanded - predictions_unfolded
        dynamic_delta = delta * mask.unsqueeze(dim=1)
        losses = torch.trapezoid(dynamic_delta * dynamic_delta, dx=0.05, dim=2)
        front_tail = torch.cumulative_trapezoid(dos_predictions**2, dx=0.05, dim=1)
        additional_error = torch.hstack(
            [
                torch.zeros(len(dos_predictions), device=device).reshape(-1, 1),
                front_tail[:, : self.extra_targets],
            ]
        )
        total_losses = losses + additional_error
        final_loss, shift = torch.min(total_losses, dim=1)
        dos_loss = torch.mean(final_loss)

        # CDOS loss
        pred_CDOS = torch.cumulative_trapezoid(dos_predictions, dx=0.05, dim=1) + cdos_start_prediction.unsqueeze(dim=1) # shape 700
        CDOS_MSE = 0
        for index_i, shift_i in enumerate(shift):
            pred_CDOS_i = pred_CDOS[index_i][shift_i + 1: shift_i + 1 + len(true_CDOS[0])]
            if len(pred_CDOS_i) < len(true_CDOS[0]):
                # Pad pred_CDOS_i with last value
                pad_size = len(true_CDOS[0]) - len(pred_CDOS_i)
                pad_values = pred_CDOS_i[-1].repeat(pad_size)
                pred_CDOS_i = torch.cat([pred_CDOS_i, pad_values], dim=0)
            CDOS_loss_i = torch.trapezoid((pred_CDOS_i - true_CDOS[index_i])**2, dx =0.05)
            CDOS_MSE += CDOS_loss_i

        CDOS_MSE = CDOS_MSE / len(systems)
        total_dos_loss = dos_loss + CDOS_MSE * self.int_weight

        # Bandgap loss
        # Get gap
        bandgap_predictions  = get_gap_differentiable(doses=dos_predictions, cdos_starts=cdos_start_prediction, n_elecs=true_n_electrons, tol=0.1, dx=0.05, temp=100.0)

        # if self.force:
        #      gap_force_predictions = torch.vstack(compute_gradient(bandgap_predictions, [system.positions for system in systems], is_training=True, destroy_graph= False))
        if self.print:
            print ("Printing Shapes")
            print ("Mask shape:", mask.shape)
            print ("DOS predictions shape:", dos_predictions.shape)
            print ("True DOS shape:", true_dos.shape)
            print ("CDOS predictions shape:", pred_CDOS.shape)
            print ("true_CDOS shape:", true_CDOS.shape)
            print("Bandgap predictions:", bandgap_predictions.shape)
            print("Bandgap targets:", true_gap.shape)
            # if self.force:
            #     print("Gap force predictions:", gap_force_predictions.shape)
            #     print("Gap force targets:", true_gapforce.shape)
            self.print = False
        
        gap_loss = torch.mean((bandgap_predictions.view(-1,1) - true_gap) ** 2)
        # if self.force:
        #     gapforce_loss = torch.mean((gap_force_predictions - true_gapforce) ** 2) 
        #     gap_loss += gapforce_loss * self.force_weight
        total_loss = total_dos_loss + gap_loss * self.gap_weight
        return total_loss
    
class PhysicalGapPDOSLoss(LossInterface):
    """
    Focused DOS loss on :py:class:`TensorMap` entries.

    :param name: key for the dos in the prediction/target dictionary.
    :param gradient: optional gradient field name.
    :param weight: weight of the loss contribution in the final aggregation.
    :param force_weight: Multiplier for the forces of the gap.
    :param force: Whether to apply the force term.
    :param reduction: reduction mode for torch loss.
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        int_weight: float,
        gap_weight: float,
        extra_targets: int,
        reduction: str,
    ):
        super().__init__(
            name,
            gradient,
            weight,
            reduction,
        )
        self.int_weight = int_weight
        self.gap_weight = gap_weight
        self.extra_targets = extra_targets # should be 300 for this exercise
        self.sigmoid = torch.nn.Sigmoid()

        self.print = True
        self.denoiser = DifferentiableDenoiser()

        
    def compute(
        self,
        model_predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        """
        gap_key = f"mtt::gap"
        CDOS_key = f"mtt::cdos"
        DOS_key = f"mtt::tdos"
        mask_key = f"mtt::mask"
        elec_key = f"mtt::nelec"
        systems, model, extra_data = extra_data
        if extra_data is None or gap_key not in extra_data:
            raise ValueError(
                f"Expected extra_data to contain TensorMap under '{gap_key}'"
            )
        if CDOS_key not in extra_data:
            raise ValueError(
                f"Expected extra_data to contain TensorMap under '{CDOS_key}'"
            )
        if mask_key not in extra_data:
            raise ValueError(
                f"Expected extra_data to contain TensorMap under '{mask_key}'"
            )
        if elec_key not in extra_data:
            raise ValueError(
                f"Expected extra_data to contain TensorMap under '{elec_key}'"
            )
        if DOS_key not in extra_data:
            raise ValueError(
                f"Expected extra_data to contain TensorMap under '{DOS_key}'"
            )

        tensor_map_pred = model_predictions[self.target] # should be the dos prediction (Normalized already)
        tensor_map_gap = extra_data[gap_key]
        tensor_map_CDOS = extra_data[CDOS_key]
        tensor_map_mask = extra_data[mask_key]
        tensor_map_elec = extra_data[elec_key]
        tensor_map_targ = extra_data[DOS_key]
        # LAST DOS VALUE IS START OF CDOS
        # There should only be one block
        denoised_dos_predictions = self.denoiser.denoise(tensor_map_pred.block().values[:,:-1])
        atomstructure_index = tensor_map_pred.block().samples['system']
        n_structures = len(systems)
        dos_predictions = torch.zeros((n_structures, denoised_dos_predictions.shape[1]), device=denoised_dos_predictions.device)
        dos_predictions.index_add_(0, atomstructure_index, denoised_dos_predictions)
        tensor_map_pred = mts.sum_over_samples(tensor_map_pred, sample_names='atom')
        cdos_start_prediction = tensor_map_pred.block().values[:,-1]
        true_gap = tensor_map_gap.block().values # Already Normalized wrt atoms
        true_dos = tensor_map_targ.block().values[:, self.extra_targets+1:] # Already normalized wrt atoms
        true_CDOS = tensor_map_CDOS.block().values[:, self.extra_targets:] # Already Normalized wrt atoms
        mask = tensor_map_mask.block().values.bool()[:, self.extra_targets:]
        true_n_electrons = tensor_map_elec.block().values.squeeze()

        # Evaluate error on DOS
        device = dos_predictions.device
        predictions_unfolded = dos_predictions.unfold(1, len(true_dos[0]), 1)
        target_expanded = true_dos[:, None, :]
        delta = target_expanded - predictions_unfolded
        dynamic_delta = delta * mask.unsqueeze(dim=1)
        losses = torch.trapezoid(dynamic_delta * dynamic_delta, dx=0.05, dim=2)
        front_tail = torch.cumulative_trapezoid(dos_predictions**2, dx=0.05, dim=1)
        additional_error = torch.hstack(
            [
                torch.zeros(len(dos_predictions), device=device).reshape(-1, 1),
                front_tail[:, : self.extra_targets],
            ]
        )
        total_losses = losses + additional_error
        final_loss, shift = torch.min(total_losses, dim=1)
        dos_loss = torch.mean(final_loss)

        # CDOS loss
        pred_CDOS = torch.cumulative_trapezoid(dos_predictions, dx=0.05, dim=1) + cdos_start_prediction.unsqueeze(dim=1) # shape 700
        CDOS_MSE = 0
        for index_i, shift_i in enumerate(shift):
            pred_CDOS_i = pred_CDOS[index_i][shift_i + 1: shift_i + 1 + len(true_CDOS[0])]
            if len(pred_CDOS_i) < len(true_CDOS[0]):
                # Pad pred_CDOS_i with last value
                pad_size = len(true_CDOS[0]) - len(pred_CDOS_i)
                pad_values = pred_CDOS_i[-1].repeat(pad_size)
                pred_CDOS_i = torch.cat([pred_CDOS_i, pad_values], dim=0)
            CDOS_loss_i = torch.trapezoid((pred_CDOS_i - true_CDOS[index_i])**2, dx =0.05)
            CDOS_MSE += CDOS_loss_i

        CDOS_MSE = CDOS_MSE / len(systems)
        total_dos_loss = dos_loss + CDOS_MSE * self.int_weight

        # Bandgap loss
        # Get gap
        bandgap_predictions  = get_gap_differentiable(doses=dos_predictions, cdos_starts=cdos_start_prediction, n_elecs=true_n_electrons, tol=0.1, dx=0.05, temp=100.0)

        # if self.force:
        #      gap_force_predictions = torch.vstack(compute_gradient(bandgap_predictions, [system.positions for system in systems], is_training=True, destroy_graph= False))
        if self.print:
            print ("Printing Shapes")
            print ("Mask shape:", mask.shape)
            print ("DOS predictions shape:", dos_predictions.shape)
            print ("True DOS shape:", true_dos.shape)
            print ("CDOS predictions shape:", pred_CDOS.shape)
            print ("true_CDOS shape:", true_CDOS.shape)
            print("Bandgap predictions:", bandgap_predictions.shape)
            print("Bandgap targets:", true_gap.shape)
            # if self.force:
            #     print("Gap force predictions:", gap_force_predictions.shape)
            #     print("Gap force targets:", true_gapforce.shape)
            self.print = False
        
        gap_loss = torch.mean((bandgap_predictions.view(-1,1) - true_gap) ** 2)
        # if self.force:
        #     gapforce_loss = torch.mean((gap_force_predictions - true_gapforce) ** 2) 
        #     gap_loss += gapforce_loss * self.force_weight
        total_loss = total_dos_loss + gap_loss * self.gap_weight
        return total_loss
    
# --- aggregator -----------------------------------------------------------------------


class LossAggregator(LossInterface):
    """
    Aggregate multiple :py:class:`LossInterface` terms with scheduled weights and
    metadata.

    :param targets: mapping from target names to :py:class:`TargetInfo`.
    :param config: per-target configuration dict.
    """

    def __init__(
        self, targets: Dict[str, TargetInfo], config: Dict[str, Dict[str, Any]]
    ):
        super().__init__(name="", gradient=None, weight=0.0, reduction="mean")
        self.losses: Dict[str, LossInterface] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

        for target_name, target_info in targets.items():
            target_config = config.get(
                target_name,
                {
                    "type": "mse",
                    "weight": 1.0,
                    "reduction": "mean",
                    "gradients": {},
                },
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
                    {
                        "type": "mse",
                        "weight": 1.0,
                        "reduction": "mean",
                    },
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
    MASKED_DOS = ("masked_dos", MaskedDOSLoss)
    GAPDOS = ("bandgap", BandgapLoss)
    HOMOLUMO = ("HOMOLUMO", HOMOLUMOLoss)
    noloss = ("noloss", NoLoss)
    gapdosloss = ("gapdosloss", GapDOSLoss)
    focuseddosloss = ("focuseddosloss", FocusedDOSLoss)
    physicalgapdosloss = ("pgapdosloss", PhysicalGapDOSLoss)
    physicalgappdosloss = ("pgappdosloss", PhysicalGapPDOSLoss)

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
