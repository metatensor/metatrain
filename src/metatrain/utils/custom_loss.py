# losses.py

from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Type

import metatensor.torch as mts
import torch
from metatensor.torch import TensorMap
from torch.nn.modules.loss import _Loss

from metatrain.utils.data import TargetInfo


class LossRegistry(ABCMeta):
    """
    Metaclass to auto-register LossBase subclasses.

    Maintains a mapping from registry_name to the subclass type.
    """

    _registry: Dict[str, Type["LossBase"]] = {}

    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        # Skip the abstract base itself
        if name != "LossBase" and issubclass(cls, LossBase):
            # Use explicit registry_name if given, else snake_case from class name
            key = getattr(cls, "registry_name", None)
            if key is None:
                key = "".join(
                    f"_{c.lower()}" if c.isupper() else c for c in name
                ).lstrip("_")
            # only register the very first class under each key
            mcs._registry.setdefault(key, cls)
        return cls

    @classmethod
    def get(cls, key: str) -> Type["LossBase"]:
        """
        Retrieve a registered LossBase subclass by its registry_name.

        :param key: The registry key for the loss.
        :return: The corresponding LossBase subclass.
        :raises KeyError: If the key is not found in the registry.
        """
        if key not in cls._registry:
            raise KeyError(
                f"Unknown loss '{key}'. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[key]


class LossBase(ABC, metaclass=LossRegistry):
    """
    Abstract base for all loss functions.

    Subclasses must implement compute(predictions, targets) -> torch.Tensor.
    """

    registry_name: str = "base"
    weight: float = 0.0
    reduction: str = "mean"
    loss_kwargs: Dict[str, Any]
    target: str = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # no-op, just so subclasses can define any signature
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

    def __call__(self, predictions: Any, targets: Any) -> torch.Tensor:
        """
        Alias to compute(), so loss instances are callable.

        :param predictions: Model outputs.
        :param targets: Ground-truth data.
        :return: Scalar loss tensor.
        """
        return self.compute(predictions, targets)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "LossBase":
        """
        Instantiate a LossBase subclass from a config dict.

        :param cfg: Keyword arguments for the loss constructor.
        :return: An instance of the loss subclass.
        """
        return cls(**cfg)


# --- specific losses ------------------------------------------------------------------


class TensorMapPointwiseLoss(LossBase):
    """
    Pointwise loss on :py:class:`TensorMap` entries using a :py:mod:`torch.nn` loss
    function.

    Extracts values or gradients, flattens them, and applies ``loss_fn``.
    """

    registry_name = "pointwise"

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
        self.target = name
        self.gradient = gradient or None
        self.weight = weight
        self.reduction = reduction

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


class TensorMapMSELoss(TensorMapPointwiseLoss):
    registry_name = "mse"

    def __init__(
        self,
        name: str,
        gradient: Optional[str] = None,
        weight: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__(
            name=name,
            gradient=gradient,
            weight=weight,
            reduction=reduction,
            loss_cls=torch.nn.MSELoss,
        )


class TensorMapMAELoss(TensorMapPointwiseLoss):
    registry_name = "mae"

    def __init__(
        self,
        name: str,
        gradient: Optional[str] = None,
        weight: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__(
            name=name,
            gradient=gradient,
            weight=weight,
            reduction=reduction,
            loss_cls=torch.nn.L1Loss,
        )


class TensorMapHuberLoss(TensorMapPointwiseLoss):
    registry_name = "huber"

    def __init__(
        self,
        name: str,
        gradient: Optional[str] = None,
        weight: float = 1.0,
        reduction: str = "mean",
        delta: float = 1.0,
    ):
        super().__init__(
            name=name,
            gradient=gradient,
            weight=weight,
            reduction=reduction,
            loss_cls=torch.nn.HuberLoss,
            delta=delta,
        )


# --- aggregator -----------------------------------------------------------------------


class LossAggregator(LossBase):
    """
    Aggregate multiple :py:class:`LossBase` terms with optional sliding weights.
    """

    registry_name = "aggregate"

    def __init__(
        self,
        targets: Dict[str, TargetInfo],
        config: Dict[str, Dict[str, Any]],
    ):
        self.sliding_weights_schedulers: Dict[str, SlidingWeightScheduler] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        cfg = config or {}

        for target_name, tm in targets.items():
            tgt_cfg = cfg.get(target_name, {})

            # main term
            MainCls = LossRegistry.get(tgt_cfg.get("type", "mse"))
            main_loss = MainCls(
                name=target_name,
                gradient=None,
                weight=tgt_cfg.get("weight", 1.0),
                reduction=tgt_cfg.get("reduction", "mean"),
            )
            sf = tgt_cfg.get("sliding_factor", None)
            self.sliding_weights_schedulers[target_name] = SlidingWeightScheduler(
                loss_fn=main_loss, sliding_factor=sf
            )

            self.metadata[target_name] = {
                "type": main_loss.registry_name,
                "weight": main_loss.weight,
                "reduction": main_loss.reduction,
                "sliding_factor": sf,
                "gradients": {},
            }

            # gradients
            all_grads = list(tm.layout[0].gradients_list())
            grad_cfgs = tgt_cfg.get("gradients", {})
            for grad_name in all_grads:
                key = f"{target_name}_grad_{grad_name}"
                gcfg = grad_cfgs.get(grad_name, {})
                GradCls = LossRegistry.get(gcfg.get("type", tgt_cfg.get("type", "mse")))
                grad_loss = GradCls(
                    name=target_name,
                    gradient=grad_name,
                    weight=gcfg.get("weight", 1.0),
                    reduction=gcfg.get("reduction", tgt_cfg.get("reduction", "mean")),
                )
                self.sliding_weights_schedulers[key] = SlidingWeightScheduler(
                    loss_fn=grad_loss,
                    sliding_factor=tgt_cfg.get("sliding_factor", None),
                )

                self.metadata[target_name]["gradients"][grad_name] = {
                    "type": grad_loss.registry_name,
                    "weight": grad_loss.weight,
                    "reduction": grad_loss.reduction,
                    "sliding_factor": sf,
                }

    def compute(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Any] = None,
    ) -> torch.Tensor:
        # initialize all on the first call
        if not all(s.initialized for s in self.sliding_weights_schedulers.values()):
            for s in self.sliding_weights_schedulers.values():
                s.initialize(targets)

        example = next(iter(predictions.values()))
        total = torch.zeros(
            (),
            dtype=example.block(0).values.dtype,
            device=example.block(0).values.device,
        )

        # sum up loss_i * weight_i / sliding_weight_i
        for sched in self.sliding_weights_schedulers.values():
            loss_val = sched.loss_fn.compute(
                predictions, targets, extra_data=extra_data
            )
            total += loss_val * sched.loss_fn.weight / sched.weight

        # update for next batch
        for sched in self.sliding_weights_schedulers.values():
            sched.update(predictions, targets)

        # # debug: print every updated sliding weight
        # print("=== DEBUG: updated sliding weights ===")
        # for key, sched in self.sliding_weights_schedulers.items():
        #     print(f"  {key:30s}: {sched.weight:.6f}")
        # print("======================================")

        return total


# --- helper classes and functions -----------------------------------------------------


class SlidingWeightScheduler:
    """
    Maintain a running Exponential Moving Average (EMA) weight for a single
    :py:class:`LossBase` term.

    Initialize from a target-mean baseline, then update via EMA.
    """

    EPS = 1e-6

    def __init__(
        self,
        loss_fn: LossBase,
        sliding_factor: Optional[float],
    ):
        self.loss_fn = loss_fn
        # collapse None to 0.0 so we never slide unless sf>0
        self.sf: float = float(sliding_factor or 0.0)
        self.weight: float = 1.0
        self.initialized = False

    def initialize(self, targets: Dict[str, TensorMap]) -> None:
        """
        Compute initial weight by comparing the target to its mean or zeros.

        :param targets: Mapping from target names to :py:class:`TensorMap`s.
        """
        if self.sf <= 0.0:
            self.weight = 1.0
        else:
            name = self.loss_fn.target
            grad = getattr(self.loss_fn, "gradient", None)

            tm = targets[name]

            # build a TensorMap baseline whose blocks hold the per‐block mean
            if grad is None:
                # mean over samples
                mean_tm = mts.mean_over_samples(tm, tm.sample_names)
                # Create a baseline TensorMap with the same structure
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
                # for a gradient‐loss, baseline is zero‐valued tensormap
                baseline = mts.zeros_like(tm)

            # now compute the loss_fn between tm and baseline
            val = self.loss_fn.compute({name: tm}, {name: baseline})
            self.weight = float(val.clamp_min(self.EPS))
        self.initialized = True

    def update(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
    ) -> None:
        """
        Update the weight with the current loss via exponential moving average.

        :param predictions: Mapping from target names to predicted
            :py:class:`TensorMap`s.
        :param targets: Mapping from target names to :py:class:`TensorMap`s.
        """

        if self.sf <= 0.0:
            return

        # compute the current error
        err = self.loss_fn.compute(predictions, targets).detach().item()
        # Exponential moving average update
        self.weight = self.sf * self.weight + (1.0 - self.sf) * err
