from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Type

import metatensor.torch as mts
import torch
from metatensor.torch import TensorMap
from torch.nn.modules.loss import _Loss

from metatrain.utils.data import TargetInfo


class LossInterface(ABC):
    """
    Abstract base for all loss functions.

    Subclasses must implement the ``compute`` method.
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
        """
        :param name: key in the predictions/targets dict to select the TensorMap.
        :param gradient: optional name of a gradient field to extract.
        :param weight: multiplicative weight (used by ScheduledLoss).
        :param reduction: reduction mode for torch losses ("mean", "sum", etc.).
        """
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
        Compute the loss.

        :param predictions: mapping from target names to :py:class:`TensorMap`.
        :param targets: mapping from target names to :py:class:`TensorMap`.
        :param extra_data: optional additional data (e.g., masks).
        :return: scalar torch.Tensor representing the loss.
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


# --- scheduler interface and implementations ------------------------------------------


class WeightScheduler(ABC):
    """
    Abstract interface for scheduling a weight for a :py:class:`LossInterface`.
    """

    initialized: bool = False

    @abstractmethod
    def initialize(
        self, loss_fn: LossInterface, targets: Dict[str, TensorMap]
    ) -> float:
        """
        Compute and return the initial weight.

        :param loss_fn: the base loss to initialize.
        :param targets: mapping of target names to :py:class:`TensorMap`.
        :return: initial weight as a float.
        """

    @abstractmethod
    def update(
        self,
        loss_fn: LossInterface,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
    ) -> float:
        """
        Update and return the new weight after a batch.

        :param loss_fn: the base loss.
        :param predictions: mapping of target names to :py:class:`TensorMap`.
        :param targets: mapping of target names to :py:class:`TensorMap`.
        :return: updated weight as a float.
        """


class EMAScheduler(WeightScheduler):
    """
    Exponential moving average scheduler for loss weights.
    """

    EPSILON = 1e-6

    def __init__(self, sliding_factor: Optional[float]) -> None:
        """
        :param sliding_factor: factor in [0,1] for EMA (0 disables scheduling).
        """
        self.sliding_factor = float(sliding_factor or 0.0)
        self.current_weight = 1.0
        self.initialized = False

    def initialize(
        self, loss_fn: LossInterface, targets: Dict[str, TensorMap]
    ) -> float:
        # If scheduling disabled, keep weight = 1.0
        if self.sliding_factor <= 0.0:
            self.current_weight = 1.0
        else:
            # Compute a baseline loss against a constant mean or zero-gradient map
            target_name = loss_fn.target
            gradient_name = getattr(loss_fn, "gradient", None)
            tensor_map_for_target = targets[target_name]

            if gradient_name is None:
                # Create a baseline TensorMap with all values = mean over samples
                mean_tensor_map = mts.mean_over_samples(
                    tensor_map_for_target, tensor_map_for_target.sample_names
                )
                baseline_tensor_map = TensorMap(
                    keys=tensor_map_for_target.keys,
                    blocks=[
                        mts.TensorBlock(
                            samples=block.samples,
                            components=block.components,
                            properties=block.properties,
                            values=torch.ones_like(block.values) * mean_block.values,
                        )
                        for block, mean_block in zip(
                            tensor_map_for_target, mean_tensor_map
                        )
                    ],
                )
            else:
                # Zero baseline for gradient-based losses
                baseline_tensor_map = mts.zeros_like(tensor_map_for_target)

            initial_loss_value = loss_fn.compute(
                {target_name: tensor_map_for_target}, {target_name: baseline_tensor_map}
            )
            self.current_weight = float(initial_loss_value.clamp_min(self.EPSILON))

        self.initialized = True
        return self.current_weight

    def update(
        self,
        loss_fn: LossInterface,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
    ) -> float:
        # If scheduling disabled, return fixed weight
        if self.sliding_factor <= 0.0:
            return self.current_weight

        # Compute the instantaneous error
        instantaneous_error = loss_fn.compute(predictions, targets).detach().item()
        # EMA update
        new_weight = (
            self.sliding_factor * self.current_weight
            + (1.0 - self.sliding_factor) * instantaneous_error
        )
        self.current_weight = max(new_weight, self.EPSILON)
        return self.current_weight


class ScheduledLoss(LossInterface):
    """
    Wrap a base :py:class:`LossInterface` with a :py:class:`WeightScheduler`.
    After each compute, the scheduler updates the loss weight.
    """

    def __init__(self, base_loss: LossInterface, weight_scheduler: WeightScheduler):
        """
        :param base_loss: underlying LossInterface to wrap.
        :param weight_scheduler: scheduler that controls the multiplier.
        """
        super().__init__(
            base_loss.target,
            base_loss.gradient,
            base_loss.weight,
            base_loss.reduction,
        )
        self.base_loss = base_loss
        self.scheduler = weight_scheduler
        self.loss_kwargs = getattr(base_loss, "loss_kwargs", {})

    def compute(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Any] = None,
    ) -> torch.Tensor:
        # Initialize scheduler on first call
        if not self.scheduler.initialized:
            self.normalization_factor = self.scheduler.initialize(
                self.base_loss, targets
            )

        # compute the raw loss using the base loss function
        raw_loss_value = self.base_loss.compute(predictions, targets, extra_data)

        # scale by the fixed weight and divide by the sliding weight
        weighted_loss_value = raw_loss_value * (
            self.base_loss.weight / self.normalization_factor
        )

        # update the sliding weight
        self.normalization_factor = self.scheduler.update(
            self.base_loss, predictions, targets
        )

        return weighted_loss_value


# --- specific losses ------------------------------------------------------------------


class BaseTensorMapLoss(LossInterface):
    """
    Backbone for pointwise losses on :py:class:`TensorMap` entries.

    Provides a compute_flattened() helper that extracts values or gradients,
    flattens them, applies an optional mask, and computes the torch loss.
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
        """
        :param name: key in the predictions/targets dict.
        :param gradient: optional gradient field name.
        :param weight: dummy here; real weighting in ScheduledLoss.
        :param reduction: reduction mode for torch loss.
        :param loss_fn: pre-instantiated torch.nn loss (e.g. MSELoss).
        """
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


class BasisContractionLoss(LossInterface):
    """
    Loss for basis contraction tasks, comparing two :py:class:`TensorMap`s
    representing density matrices.
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        reduction: str,
    ):
        """
        :param name: key in the predictions/targets dict.
        :param gradient: must be None (not implemented).
        :param weight: dummy here; real weighting in ScheduledLoss.
        :param reduction: whatever torch accepts.
        """
        if gradient is not None:
            raise ValueError("BasisContractionLoss does not support gradients")
        super().__init__(name, gradient, weight, reduction)

    def compute(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Compute the Frobenius norm between predicted and target density matrices.

        :param predictions: mapping of names to :py:class:`TensorMap`.
        :param targets: ignored.
        :param extra_data: actual targets.
        :return: scalar torch.Tensor loss.
        """
        tensor_map_pred = predictions[self.target]
        name_in_extra_data = self.target.replace("desired", "total")
        tensor_map_targ = extra_data[name_in_extra_data]
        batch_size = tensor_map_pred[0].shape[0]
        contracted_size = int(tensor_map_pred[0].properties.values[:, 1].max() + 1)
        total_basis_size = int(tensor_map_targ[0].properties.values[:, 1].max() + 1)

        concat = torch.concatenate(
            [
                block.values.reshape(*block.shape[:2], -1, contracted_size)
                for block in tensor_map_pred
            ],
            dim=1,
        ).reshape(batch_size, -1, contracted_size)

        reconstructed = torch.bmm(concat, concat.transpose(1, 2))
        target = torch.concatenate(
            [
                block.values.reshape(*block.shape[:2], -1, contracted_size)
                for block in tensor_map_targ
            ],
            dim=1,
        ).reshape(batch_size, -1, total_basis_size)
        assert reconstructed.shape == target.shape

        return torch.functional.F.mse_loss(
            reconstructed, target, reduction=self.reduction
        )


# --- aggregator -----------------------------------------------------------------------


class LossAggregator(LossInterface):
    """
    Aggregate multiple :py:class:`LossInterface` terms with scheduled weights and
    metadata.
    """

    def __init__(
        self, targets: Dict[str, TargetInfo], config: Dict[str, Dict[str, Any]]
    ):
        """
        :param targets: mapping from target names to :py:class:`TargetInfo`.
        :param config: per-target configuration dict.
        """
        super().__init__(name="", gradient=None, weight=0.0, reduction="mean")
        self.scheduled_losses: Dict[str, ScheduledLoss] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

        for target_name, target_info in targets.items():
            target_config = config[target_name]

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
                        "sliding_factor",
                        "gradients",
                    )
                },
            )
            ema_scheduler = EMAScheduler(target_config["sliding_factor"])
            scheduled_main_loss = ScheduledLoss(base_loss, ema_scheduler)
            self.scheduled_losses[target_name] = scheduled_main_loss
            self.metadata[target_name] = {
                "type": target_config["type"],
                "weight": base_loss.weight,
                "reduction": base_loss.reduction,
                "sliding_factor": target_config["sliding_factor"],
                "gradients": {},
            }
            for pname, pval in target_config.items():
                if pname not in (
                    "type",
                    "weight",
                    "reduction",
                    "sliding_factor",
                    "gradients",
                ):
                    self.metadata[target_name][pname] = pval

            # Create gradient-based losses
            gradient_config = target_config["gradients"]
            for gradient_name in target_info.layout[0].gradients_list():
                gradient_key = f"{target_name}_grad_{gradient_name}"

                gradient_specific_config = gradient_config[gradient_name]

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
                            "sliding_factor",
                            "gradients",
                        )
                    },
                )
                ema_scheduler_for_grad = EMAScheduler(target_config["sliding_factor"])
                scheduled_grad_loss = ScheduledLoss(grad_loss, ema_scheduler_for_grad)
                self.scheduled_losses[gradient_key] = scheduled_grad_loss
                self.metadata[target_name]["gradients"][gradient_name] = {
                    "type": gradient_specific_config["type"],
                    "weight": grad_loss.weight,
                    "reduction": grad_loss.reduction,
                    "sliding_factor": target_config["sliding_factor"],
                }
                for pname, pval in gradient_specific_config.items():
                    if pname not in (
                        "type",
                        "weight",
                        "reduction",
                        "sliding_factor",
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
        """
        # Initialize a zero tensor matching the dtype and device of the first block
        first_tensor_map = next(iter(predictions.values()))
        first_block = first_tensor_map.block(first_tensor_map.keys[0])
        total_loss = torch.zeros(
            (), dtype=first_block.values.dtype, device=first_block.values.device
        )

        # Sum each scheduled term that has a matching prediction
        for scheduled_term in self.scheduled_losses.values():
            if scheduled_term.target not in predictions:
                continue
            total_loss = total_loss + scheduled_term.compute(
                predictions, targets, extra_data
            )

        return total_loss


class LossType(Enum):
    """
    Enumeration of available loss types and their implementing classes.
    """

    MSE = ("mse", TensorMapMSELoss)
    MAE = ("mae", TensorMapMAELoss)
    HUBER = ("huber", TensorMapHuberLoss)
    MASKED_MSE = ("masked_mse", TensorMapMaskedMSELoss)
    MASKED_MAE = ("masked_mae", TensorMapMaskedMAELoss)
    MASKED_HUBER = ("masked_huber", TensorMapMaskedHuberLoss)
    POINTWISE = ("pointwise", BaseTensorMapLoss)
    MASKED_POINTWISE = ("masked_pointwise", MaskedTensorMapLoss)
    BASIS_CONTRACTION = ("basis_contraction", BasisContractionLoss)

    def __init__(self, key: str, cls: Type[LossInterface]):
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

        :raises ValueError: if the key is not valid.
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
    """
    Factory to instantiate a concrete :py:class:`LossInterface` given its string key.

    :param loss_type: string key matching one of the members of :py:class:`LossType`.
    :param name: target name for the loss.
    :param gradient: gradient name, if present.
    :param weight: weight for the loss contribution.
    :param reduction: reduction mode for the torch loss.
    :param extra_kwargs: additional hyperparameters specific to the loss type.
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
