# losses.py

from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Type

import torch
from metatensor.torch import TensorMap


class LossRegistry(ABCMeta):
    """Metaclass that auto-registers every LossBase subclass."""

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
            if key in mcs._registry:
                raise KeyError(f"Loss '{key}' already registered")
            mcs._registry[key] = cls
        return cls

    @classmethod
    def get(cls, key: str) -> Type["LossBase"]:
        if key not in cls._registry:
            raise KeyError(
                f"Unknown loss '{key}'. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[key]


class LossBase(ABC, metaclass=LossRegistry):
    """All losses implement compute(predictions, batch) -> scalar Tensor."""

    registry_name: str = "base"
    weight: float = 0.0

    @abstractmethod
    def compute(self, predictions: Any, batch: Any) -> torch.Tensor: ...

    def __call__(self, predictions: Any, batch: Any) -> torch.Tensor:
        return self.compute(predictions, batch)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "LossBase":
        return cls(**cfg)


# --- specific losses ------------------------------------------------------------------


class TensorMapMSELoss(LossBase):
    registry_name = "mse"

    def __init__(self, name: str, weight: float = 1.0, reduction: str = "mean",  gradient_weights: Optional[Dict[str, float]] = None, sliding_factor: Optional[float] = None):
        self.target = name

        if gradient_weights is None:
            gradient_weights = {}
        losses = {}
        losses["values"] = torch.nn.MSELoss(reduction=reduction)
        for key in gradient_weights.keys():
            losses[key] = torch.nn.L1Loss(reduction=reduction)        
        self.losses = losses
        
        self.weight = weight
        self.gradient_weights = gradient_weights
        self.sliding_factor = sliding_factor
        self.sliding_weights: Optional[Dict[str, TensorMap]] = None

    def compute(
        self, predictions: Dict[str, TensorMap], targets: Dict[str, TensorMap]
    ) -> torch.Tensor:
        
        pred_tm = predictions[self.target]
        tgt_tm = targets[self.target]

        ## IMPLEMENT CHECK ROUTINE

        # First time the function is called: compute the sliding weights only
        # from the targets (if they are enabled)
        if self.sliding_factor is not None and self.sliding_weights is None:
            self.sliding_weights = get_sliding_weights(
                self.losses,
                self.sliding_factor,
                tgt_tm,
            )

        loss = torch.zeros(
            (),
            dtype=pred_tm.block(0).values.dtype,
            device=pred_tm.block(0).values.device,
        )

        for key in pred_tm.keys:

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

            ## loss on the main values with the sliding weight
            loss += (
                self.weight * self.losses["values"](values_1, values_2) / sliding_weight
            )

            for gradient_name in block_2.gradients_list():    ## check if this works
                gradient_weight = self.gradient_weights[gradient_name]
                values_1 = block_1.gradient(gradient_name).values
                values_2 = block_2.gradient(gradient_name).values

                sliding_weight = (
                    1.0
                    if self.sliding_weights is None
                    else self.sliding_weights.get(gradient_name, 1.0)
                )
                loss += (
                    gradient_weight
                    * self.losses[gradient_name](values_1, values_2)
                    / sliding_weight
                )

        # update sliding weights 
        if self.sliding_factor is not None:
            self.sliding_weights = get_sliding_weights(
                self.losses,
                self.sliding_factor,
                tgt_tm,
                pred_tm,
                self.sliding_weights,
            )

        return loss


class TensorMapMAELoss(LossBase):
    registry_name = "mae"

    def __init__(self, name: str, weight: float = 1.0, reduction: str = "mean"):
        self.target = name
        self.weight = weight
        self.loss_fn = torch.nn.L1Loss(reduction=reduction)

    def compute(
        self, predictions: Dict[str, TensorMap], targets: Dict[str, TensorMap]
    ) -> torch.Tensor:
        pred_tm = predictions[self.target]
        tgt_tm = targets[self.target]
        loss = torch.zeros(
            (),
            dtype=pred_tm.block(0).values.dtype,
            device=pred_tm.block(0).values.device,
        )
        for key in pred_tm.keys:
            loss = loss + self.loss_fn(
                pred_tm.block(key).values, tgt_tm.block(key).values
            )
        return self.weight * loss


class TensorMapHuberLoss(LossBase):
    registry_name = "huber"

    def __init__(
        self,
        name: str,
        weight: float = 1.0,
        reduction: str = "mean",
        delta: float = 1.0,
    ):
        self.target = name
        self.weight = weight
        self.loss_fn = torch.nn.HuberLoss(reduction=reduction, delta=delta)

    def compute(
        self, predictions: Dict[str, TensorMap], targets: Dict[str, TensorMap]
    ) -> torch.Tensor:
        pred_tm = predictions[self.target]
        tgt_tm = targets[self.target]
        loss = torch.zeros(
            (),
            dtype=pred_tm.block(0).values.dtype,
            device=pred_tm.block(0).values.device,
        )
        for key in pred_tm.keys:
            loss = loss + self.loss_fn(
                pred_tm.block(key).values, tgt_tm.block(key).values
            )
        return self.weight * loss


# --- aggregator -----------------------------------------------------------------------
class LossAggregator(LossBase):

    registry_name = "aggregate"

    def __init__(
        self,
        target_names: List[str],
        config: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        self.config = config or {}
        self.losses: Dict[str, LossBase] = {}

        # for target, cfg in self.config.items():
        for target in target_names:
            cfg = self.config.get(target, {})
            loss_type = cfg.get("type", "mse")
            weight = cfg.get("weight", 1.0)

            # build kwargs for sub-loss
            params = {k: v for k, v in cfg.items() if k not in ("type", "weight")}
            params.update({"name": target, "weight": weight})

            LossCls = LossRegistry.get(loss_type)
            self.losses[target] = LossCls.from_config(params)

    def compute(
        self, predictions: Dict[str, TensorMap], targets: Dict[str, TensorMap]
    ) -> torch.Tensor:
        
        # get device/dtype from first TensorMap
        first_tm = next(iter(predictions.values()))
        total = torch.zeros(
            (),
            dtype=first_tm.block(0).values.dtype,
            device=first_tm.block(0).values.device,
        )

        for target in predictions:
            self.losses[target](predictions, targets)
            
            total = total + 

        return total


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