# losses.py

from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Type, List

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

    def __init__(self, name: str, weight: float = 1.0, reduction: str = "mean"):
        self.target = name
        self.weight = weight
        self.loss_fn = torch.nn.MSELoss(reduction=reduction)

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


class LossAggregator(LossBase):  # TensorMapDictLoss
    """Aggregate per-target sub-losses according to a config dict.

    Config example:
      {
        'energy': {'type':'mse',   'weight':1.0},
        'forces': {'type':'mse',   'weight':1.0},
        'dos':    {'type':'super_dos_loss', 'weight':1.0, 'mask_names':'foo'}
      }
    Unspecified targets default to type='mse', weight=1.0.
    """

    registry_name = "aggregate"

    def __init__(
        self,
        target_names: List[str],
        config: Optional[Dict[str, Dict[str, Any]]] = None,
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
            total = total + self.losses[target](predictions, targets)

        return total
