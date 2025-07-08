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

    Subclasses must implement compute(predictions, targets) -> torch.Tensor.
    """

    weight: float
    reduction: str
    loss_kwargs: Dict[str, Any]
    target: str
    gradient: Optional[str]

    def __init__(
        self,
        name: str,
        gradient: Optional[str] = None,
        weight: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        self.target = name
        self.gradient = gradient
        self.weight = weight
        self.reduction = reduction
        self.loss_kwargs = {}
        super().__init__()

    @abstractmethod
    def compute(
        self, predictions: Any, targets: Any, extra_data: Optional[Any] = None
    ) -> torch.Tensor:
        """
        Compute the loss given predictions and targets.

        :param predictions: Model outputs.
        :param targets: Ground-truth data.
        :param extra_data: Optional additional data for loss computation.
        :return: Scalar loss tensor.
        """
        ...

    def __call__(
        self, predictions: Any, targets: Any, extra_data: Optional[Any] = None
    ) -> torch.Tensor:
        """
        Alias to compute(), so loss instances are callable.

        :param predictions: Model outputs.
        :param targets: Ground-truth data.
        :return: Scalar loss tensor.
        """
        return self.compute(predictions, targets, extra_data)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "LossInterface":
        """
        Instantiate a LossBase subclass from a config dict.

        :param cfg: Keyword arguments for the loss constructor.
        :return: An instance of the loss subclass.
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
        """Compute and return the initial weight."""

    @abstractmethod
    def update(
        self,
        loss_fn: LossInterface,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
    ) -> float:
        """Update and return the new weight after a batch."""


class EMAScheduler(WeightScheduler):
    """
    Exponential moving average scheduler for loss weights.
    """

    EPS = 1e-6
    initialized: bool = False

    def __init__(self, sliding_factor: Optional[float]) -> None:
        self.sf: float = float(sliding_factor or 0.0)
        self.weight: float = 1.0
        self.initialized: bool = False

    def initialize(
        self, loss_fn: LossInterface, targets: Dict[str, TensorMap]
    ) -> float:
        if self.sf <= 0.0:
            self.weight = 1.0
        else:
            name = loss_fn.target
            grad = getattr(loss_fn, "gradient", None)
            tm = targets[name]
            if grad is None:
                mean_tm = mts.mean_over_samples(tm, tm.sample_names)
                baseline = TensorMap(
                    keys=tm.keys,
                    blocks=[
                        mts.TensorBlock(
                            samples=block.samples,
                            components=block.components,
                            properties=block.properties,
                            values=torch.ones_like(block.values) * mean_block.values,
                        )
                        for block, mean_block in zip(tm, mean_tm)
                    ],
                )
            else:
                baseline = mts.zeros_like(tm)

            val = loss_fn.compute({name: tm}, {name: baseline})
            self.weight = float(val.clamp_min(self.EPS))
        self.initialized = True
        return self.weight

    def update(
        self,
        loss_fn: LossInterface,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
    ) -> float:
        if self.sf <= 0.0:
            return self.weight
        err = loss_fn.compute(predictions, targets).detach().item()
        new_weight = self.sf * self.weight + (1.0 - self.sf) * err
        self.weight = max(new_weight, self.EPS)
        return self.weight


class ScheduledLoss(LossInterface):
    """
    Wrap a base :py:class:`LossInterface` with a :py:class:`WeightScheduler`.

    After each compute, the scheduler updates the loss weight.
    """

    def __init__(
        self,
        base_loss: LossInterface,
        scheduler: WeightScheduler,
    ):
        # Delegate attributes
        super().__init__(
            base_loss.target,
            base_loss.gradient,
            base_loss.weight,
            base_loss.reduction,
        )
        self.base = base_loss
        self.scheduler = scheduler
        self.loss_kwargs = getattr(base_loss, "loss_kwargs", {})

    def compute(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Any] = None,
    ) -> torch.Tensor:
        # Initialize the base loss weight on the first call
        if not self.scheduler.initialized:
            self.sliding_weight = self.scheduler.initialize(self.base, targets)

        # compute the raw loss using the base loss function
        raw_loss = self.base.compute(predictions, targets, extra_data)

        # scale by the fixed weight and divide by the sliding weight
        weighted_loss = raw_loss * (self.base.weight / self.sliding_weight)

        # update the sliding weight
        self.sliding_weight = self.scheduler.update(self.base, predictions, targets)

        return weighted_loss


# --- specific losses ------------------------------------------------------------------


class TensorMapPointwiseLoss(LossInterface):
    """
    Pointwise loss on :py:class:`TensorMap` entries using a :py:mod:`torch.nn` loss
    function.

    Extracts values or gradients, flattens them, and applies ``loss_fn``.
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str] = None,
        weight: float = 1.0,
        reduction: str = "mean",
        *,
        loss_cls: Type[_Loss],
        **loss_kwargs,
    ):
        super().__init__(name, gradient, weight, reduction)
        self.loss_kwargs = loss_kwargs
        params = {"reduction": reduction, **loss_kwargs}
        self.loss_fn = loss_cls(**params)

    def compute(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Gather and flatten target and prediction blocks, then compute loss.

        :param predictions: Mapping from target names to TensorMaps.
        :param targets: Mapping from target names to TensorMaps.
        :param extra_data: Additional data for loss computation [ignored].
        :return: Scalar loss tensor.
        """
        del extra_data  # Unused, but kept for compatibility
        pred_parts = []
        targ_parts = []

        def grab(block, grad):
            # if grad is  None, take values; else take that gradient
            if grad is not None:
                return block.gradient(grad).values.reshape(-1)
            else:
                return block.values.reshape(-1)

        pred_tensor = predictions[self.target]
        targ_tensor = targets[self.target]

        for key in pred_tensor.keys:
            pred_block = pred_tensor.block(key)
            targ_block = targ_tensor.block(key)
            pred_parts.append(grab(pred_block, self.gradient))
            targ_parts.append(grab(targ_block, self.gradient))

        # concatenate all parts into a single tensor
        all_pred = torch.cat(pred_parts)
        all_targ = torch.cat(targ_parts)

        return self.loss_fn(all_pred, all_targ)


class TensorMapMaskedPointwiseLoss(TensorMapPointwiseLoss):
    """
    Pointwise loss on :py:class:`TensorMap` entries using a :py:mod:`torch.nn` loss
    function.

    Extracts values or gradients, flattens them, and applies ``loss_fn``.
    """

    def compute(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Any] = None,
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

        pred_parts = []
        targ_parts = []

        def grab(block, grad):
            # if grad is  None, take values; else take that gradient
            if grad is not None:
                return block.gradient(grad).values.reshape(-1)
            else:
                return block.values.reshape(-1)

        pred_tensor = predictions[self.target]
        targ_tensor = targets[self.target]
        if extra_data is None or self.target + "_mask" not in extra_data:
            raise ValueError(
                f"Expected extra_data to contain data field "
                f"'{self.target}_mask' for masking the loss."
            )
        mask_tensor = extra_data[self.target + "_mask"]

        for key in pred_tensor.keys:
            pred_block = pred_tensor.block(key)
            targ_block = targ_tensor.block(key)
            mask_block = mask_tensor.block(key)
            grabbed_mask = grab(mask_block, self.gradient)
            assert grabbed_mask.dtype == torch.bool, (
                f"Expected mask tensor to have boolean dtype, got {grabbed_mask.dtype}"
            )
            pred_parts.append(grab(pred_block, self.gradient)[grabbed_mask])
            targ_parts.append(grab(targ_block, self.gradient)[grabbed_mask])

        # concatenate all parts into a single tensor
        all_pred = torch.cat(pred_parts)
        all_targ = torch.cat(targ_parts)

        return self.loss_fn(all_pred, all_targ)


class TensorMapMSELoss(TensorMapPointwiseLoss):
    def __init__(
        self,
        name: str,
        gradient: Optional[str] = None,
        weight: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__(
            name,
            gradient,
            weight,
            reduction,
            loss_cls=torch.nn.MSELoss,
        )


class TensorMapMAELoss(TensorMapPointwiseLoss):
    def __init__(
        self,
        name: str,
        gradient: Optional[str] = None,
        weight: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__(
            name,
            gradient,
            weight,
            reduction,
            loss_cls=torch.nn.L1Loss,
        )


class TensorMapHuberLoss(TensorMapPointwiseLoss):
    def __init__(
        self,
        name: str,
        gradient: Optional[str] = None,
        weight: float = 1.0,
        reduction: str = "mean",
        delta: float = 1.0,
    ):
        super().__init__(
            name,
            gradient,
            weight,
            reduction,
            loss_cls=torch.nn.HuberLoss,
            delta=delta,
        )


class TensorMapMaskedMSELoss(TensorMapMaskedPointwiseLoss):
    def __init__(
        self,
        name: str,
        gradient: Optional[str] = None,
        weight: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__(
            name,
            gradient,
            weight,
            reduction,
            loss_cls=torch.nn.MSELoss,
        )


class TensorMapMaskedMAELoss(TensorMapMaskedPointwiseLoss):
    def __init__(
        self,
        name: str,
        gradient: Optional[str] = None,
        weight: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__(
            name,
            gradient,
            weight,
            reduction,
            loss_cls=torch.nn.L1Loss,
        )


class TensorMapMaskedHuberLoss(TensorMapMaskedPointwiseLoss):
    def __init__(
        self,
        name: str,
        gradient: Optional[str] = None,
        weight: float = 1.0,
        reduction: str = "mean",
        delta: float = 1.0,
    ):
        super().__init__(
            name,
            gradient,
            weight,
            reduction,
            loss_cls=torch.nn.HuberLoss,
            delta=delta,
        )


# --- aggregator -----------------------------------------------------------------------


class LossAggregator(LossInterface):
    """
    Aggregate multiple :py:class:`LossInterface` terms with scheduled weights and
    metadata.
    """

    def __init__(
        self,
        targets: Dict[str, TargetInfo],
        config: Dict[str, Dict[str, Any]],
    ):
        super().__init__(name="", gradient=None, weight=0.0, reduction="mean")
        # Scheduled losses and metadata stored by term key
        self.losses: Dict[str, ScheduledLoss] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

        for name, tm_info in targets.items():
            cfg = config.get(name, {})

            # Main loss
            base_loss = create_loss(
                cfg.get("type", "mse"),
                name=name,
                gradient=None,
                weight=cfg.get("weight", 1.0),
                reduction=cfg.get("reduction", "mean"),
                **{
                    k: v
                    for k, v in cfg.items()
                    if k not in ("type", "weight", "reduction", "sliding_factor")
                },
            )
            ema = EMAScheduler(cfg.get("sliding_factor", None))
            sched_loss = ScheduledLoss(base_loss, ema)
            self.losses[name] = sched_loss
            self.metadata[name] = {
                "type": cfg.get("type", "mse"),
                "weight": base_loss.weight,
                "reduction": base_loss.reduction,
                "sliding_factor": cfg.get("sliding_factor", None),
                "gradients": {},
            }

            # Gradient losses
            grad_cfgs = cfg.get("gradients", {})
            for grad_name in tm_info.layout[0].gradients_list():
                key = f"{name}_grad_{grad_name}"
                gcfg = grad_cfgs.get(grad_name, {})
                grad_loss = create_loss(
                    gcfg.get("type", cfg.get("type", "mse")),
                    name=name,
                    gradient=grad_name,
                    weight=gcfg.get("weight", 1.0),
                    reduction=gcfg.get("reduction", cfg.get("reduction", "mean")),
                    **{
                        k: v
                        for k, v in gcfg.items()
                        if k not in ("type", "weight", "reduction", "sliding_factor")
                    },
                )
                ema_grad = EMAScheduler(cfg.get("sliding_factor", None))
                sched_grad = ScheduledLoss(grad_loss, ema_grad)
                self.losses[key] = sched_grad
                self.metadata[name]["gradients"][grad_name] = {
                    "type": gcfg.get("type", cfg.get("type", "mse")),
                    "weight": grad_loss.weight,
                    "reduction": grad_loss.reduction,
                    "sliding_factor": cfg.get("sliding_factor", None),
                }

    def compute(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Any] = None,
    ) -> torch.Tensor:
        example = next(iter(predictions.values()))
        total = torch.zeros(
            (),
            dtype=example.block(0).values.dtype,
            device=example.block(0).values.device,
        )
        for term in self.losses.values():
            if term.target not in predictions:
                continue
            total = total + term.compute(predictions, targets, extra_data)
        return total


# ----- enum and factory ---------------------------------------------------------------


class LossType(Enum):
    """
    Enum to represent available loss types.
    """

    MSE = ("mse", TensorMapMSELoss)
    MAE = ("mae", TensorMapMAELoss)
    HUBER = ("huber", TensorMapHuberLoss)
    MASKED_MSE = ("masked_mse", TensorMapMaskedMSELoss)
    MASKED_MAE = ("masked_mae", TensorMapMaskedMAELoss)
    MASKED_HUBER = ("masked_huber", TensorMapMaskedHuberLoss)
    POINTWISE = ("pointwise", TensorMapPointwiseLoss)
    MASKED_POINTWISE = ("masked_pointwise", TensorMapMaskedPointwiseLoss)

    def __init__(self, key: str, cls: Type[LossInterface]):
        self._key = key
        self._cls = cls

    @property
    def key(self) -> str:
        return self._key

    @property
    def cls(self) -> Type[LossInterface]:
        return self._cls

    @classmethod
    def from_key(cls, key: str) -> "LossType":
        for lt in cls:
            if lt.key == key:
                return lt
        valid = ", ".join(lt.key for lt in cls)
        raise ValueError(f"Unknown loss '{key}'. Valid types: {valid}")


def create_loss(
    loss_type: str,
    *,
    name: str,
    gradient: Optional[str] = None,
    weight: float = 1.0,
    reduction: str = "mean",
    **extra_kwargs: Any,
) -> LossInterface:
    """
    Factory to instantiate a concrete ``LossInterface`` given its string key.
    """
    lt = LossType.from_key(loss_type)
    try:
        return lt.cls(
            name=name,
            gradient=gradient,
            weight=weight,
            reduction=reduction,
            **extra_kwargs,
        )
    except TypeError as e:
        # catch wrong arguments to the loss constructor
        raise TypeError(f"Error constructing loss '{loss_type}': {e}") from e
