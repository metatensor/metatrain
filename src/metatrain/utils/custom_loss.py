# losses.py

from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Type

import torch
from metatensor.torch import TensorMap
from torch.nn.modules.loss import _Loss


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
            # only register the very first class under each key
            mcs._registry.setdefault(key, cls)
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
    loss_kwargs: Dict[str, Any]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # no-op, just so subclasses can define any signature
        self.loss_kwargs = {}
        super().__init__()

    @abstractmethod
    def compute(self, predictions: Any, batch: Any) -> torch.Tensor: ...

    def __call__(self, predictions: Any, batch: Any) -> torch.Tensor:
        return self.compute(predictions, batch)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "LossBase":
        return cls(**cfg)


# --- specific losses ------------------------------------------------------------------


# class TensorMapMSELossWithSliding(LossBase):
#     registry_name = "mse_sliding"

#     def __init__(
#         self,
#         name: str,
#         weight: float = 1.0,
#         reduction: str = "mean",
#         *,
#         gradient_weights: Optional[Dict[str, float]] = None,
#         sliding_factor: Optional[float] = None,
#     ):
#         self.target = name

#         if gradient_weights is None:
#             gradient_weights = {}
#         loss_fns = {}
#         loss_fns[self.target] = torch.nn.MSELoss(reduction=reduction)
#         for key in gradient_weights.keys():
#             loss_fns[f"{self.target}_{key}"] = torch.nn.MSELoss(reduction=reduction)
#         self.loss_fns = loss_fns

#         self.weight = weight
#         self.gradient_weights = gradient_weights
#         self.sliding_factor = sliding_factor
#         self.sliding_weights: Optional[Dict[str, TensorMap]] = None

#     def compute(
#         self, predictions: Dict[str, TensorMap], targets: Dict[str, TensorMap]
#     ) -> torch.Tensor:

#         pred_tm = predictions[self.target]
#         tgt_tm = targets[self.target]

#         # IMPLEMENT CHECK ROUTINE

#         # First time the function is called: compute the sliding weights only
#         # from the targets (if they are enabled)
#         if self.sliding_factor is not None and self.sliding_weights is None:
#             self.sliding_weights = get_sliding_weights(
#                 self.losses,
#                 self.sliding_factor,
#                 tgt_tm,
#             )

#         loss = torch.zeros(
#             (),
#             dtype=pred_tm.block(0).values.dtype,
#             device=pred_tm.block(0).values.device,
#         )

#         for key in pred_tm.keys:

#             pred_block = pred_tm.block(key)
#             targ_block = tgt_tm.block(key)
#             pred_values = pred_block.values
#             targ_values = targ_block.values

#             # sliding weights: default to 1.0 if not used/provided for this target
#             sliding_weight = (
#                 1.0
#                 if self.sliding_weights is None
#                 else self.sliding_weights.get("values", 1.0)
#             )

#             # loss on the main values with the sliding weight
#             loss += (
#                 self.weight
#                 * self.losses["values"](pred_values, targ_values)
#                 / sliding_weight
#             )

#             for gradient_name in targ_block.gradients_list():  # check if this works

#                 target_and_gradient = f"{self.target}_{gradient_name}"

#                 gradient_weight = self.gradient_weights.get(target_and_gradient, 1.0)

#                 pred_values = pred_block.gradient(gradient_name).values
#                 targ_values = targ_block.gradient(gradient_name).values

#                 loss += gradient_weight * self.losses[target_and_gradient](
#                     pred_values, targ_values
#                 )

#                 # sliding_weight = (
#                 #     1.0
#                 #     if self.sliding_weights is None
#                 #     else self.sliding_weights.get(target_and_gradient, 1.0)
#                 # )
#                 # loss += (
#                 #     gradient_weight
#                 #     * self.losses[target_and_gradient](pred_values, targ_values)
#                 #     / sliding_weight
#                 # )

#         # update sliding weights
#         if self.sliding_factor is not None:
#             self.sliding_weights = get_sliding_weights(
#                 self.losses,
#                 self.sliding_factor,
#                 tgt_tm,
#                 pred_tm,
#                 self.sliding_weights,
#             )

#         return loss


class TensorMapPointwiseLoss(LossBase):
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

        self.loss_kwargs = loss_kwargs

        params = {"reduction": reduction, **loss_kwargs}
        print(params)
        self.loss_fn = loss_cls(**params)

    def compute(
        self, predictions: Dict[str, TensorMap], targets: Dict[str, TensorMap]
    ) -> torch.Tensor:
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
    registry_name = "aggregate"

    def __init__(
        self,
        targets: Dict[str, TensorMap],  # TODO: actually targetinfo
        config: Dict[str, Dict[str, Any]],
    ):
        """
        config:
          {
            "energy": {
                "type":      "mse",
                "weight":    1.0,
                "reduction": "mean",        # optional, default "mean"
                "gradients": {
                  "positions": { "type": "mse", "weight": 1.0 },
                  "cell":      { "type": "mse", "weight": 1.0 },
                }
                "sliding_factor": 0.5,      # optional, default "None"
            },
            "another_target": { ... },
            ...
          }
        """
        self.loss_fns: Dict[str, LossBase] = {}
        self.grad_loss_fns: Dict[str, Dict[str, LossBase]] = {}
        self.sliding_factors: Dict[str, Optional[float]] = {}
        self.sliding_weights: Dict[str, float] = {}

        cfg = config or {}

        for target_name, target in targets.items():
            # main loss on the target
            tgt_cfg = cfg.get(target_name, {})
            main_type = tgt_cfg.get("type", "mse")
            main_weight = tgt_cfg.get("weight", 1.0)
            main_reduction = tgt_cfg.get("reduction", "mean")

            MainCls = LossRegistry.get(main_type)
            self.loss_fns[target_name] = MainCls(
                name=target_name,
                gradient=None,  # not a gradient
                weight=main_weight,
                reduction=main_reduction,
            )

            # loss_fns on gradients
            all_grads = list(target.layout[0].gradients_list())
            grad_cfgs = tgt_cfg.get("gradients", {})

            for grad_name in all_grads:
                self.grad_loss_fns[target_name] = {}

                key_grad = f"{target_name}_grad_{grad_name}"

                grad_cfg = grad_cfgs.get(grad_name, {})
                grad_type = grad_cfg.get("type", main_type)
                grad_weight = grad_cfg.get("weight", 1.0)
                grad_reduction = grad_cfg.get("reduction", main_reduction)

                GradCls = LossRegistry.get(grad_type)

                self.grad_loss_fns[target_name][key_grad] = GradCls(
                    name=target_name,
                    gradient=grad_name,
                    weight=grad_weight,
                    reduction=grad_reduction,
                )

            self.sliding_factors[target_name] = tgt_cfg.get(
                "sliding_factor", None
            )  # TODO: check type

            # compute initial sliding weights if sliding factor given
            sf = self.sliding_factors[target_name]
            if sf is not None:
                cur_sliding_weights = get_sliding_weights(
                    target_name,
                    self.loss_fns[target_name],
                    self.grad_loss_fns[target_name],
                    sf,
                    targets,
                )
                for k, v in cur_sliding_weights.items():
                    self.sliding_weights[k] = v

            # default the sliding weights to 0 (this would never be updated, given
            # sliding_factor = None)
            else:
                self.sliding_weights[target_name] = 1.0
                for k in self.grad_loss_fns[target_name].keys():
                    self.sliding_weights[k] = 1.0

    def compute(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
    ) -> torch.Tensor:
        # start from zero on the right device/dtype
        example = next(iter(predictions.values()))
        out = torch.zeros(
            (),
            dtype=example.block(0).values.dtype,
            device=example.block(0).values.device,
        )  # TODO: allow access to individual loss contributions

        for target_name, loss_fn in self.loss_fns.items():
            out = (
                out
                + loss_fn.compute(predictions, targets)
                * loss_fn.weight
                / self.sliding_weights[target_name]
            )

            for grad_loss_name, grad_loss_fn in self.grad_loss_fns[target_name].items():
                out = (
                    out
                    + grad_loss_fn.compute(predictions, targets)
                    * grad_loss_fn.weight
                    / self.sliding_weights[grad_loss_name]
                )

            # update sliding weights
            sf = self.sliding_factors[target_name]
            if sf is not None:
                self.sliding_weights = get_sliding_weights(
                    target_name,
                    loss_fn,
                    self.grad_loss_fns[target_name],
                    sf,
                    targets[target_name],
                    predictions[target_name],
                    self.sliding_weights,
                )

        return out


# --- helper functions -----------------------------------------------------------------


def get_sliding_weights(
    target_name: str,
    loss_fn: _Loss,
    grad_loss_fns: Dict[str, _Loss],
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
            sliding_weights[target_name] = (
                loss_fn(values, values.mean() * torch.ones_like(values)) + 1e-6
            )
            for gradient_name, gradient_block in block.gradients():
                grad_loss_name = f"{target_name}_grad_{gradient_name}"
                values = gradient_block.values
                sliding_weights[grad_loss_name] = grad_loss_fns[grad_loss_name](
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
                sliding_weights[target_name] = (
                    sliding_factor * previous_sliding_weights[target_name]
                    + (1 - sliding_factor)
                    * loss_fn(predictions_values, target_values).detach()
                )
                for gradient_name, gradient_block in target_block.gradients():
                    grad_loss_name = f"{target_name}_grad_{gradient_name}"
                    target_values = gradient_block.values
                    predictions_values = predictions_block.gradient(
                        gradient_name
                    ).values
                    sliding_weights[grad_loss_name] = (
                        sliding_factor * previous_sliding_weights[grad_loss_name]
                        + (1 - sliding_factor)
                        * grad_loss_fns[grad_loss_name](
                            predictions_values, target_values
                        ).detach()
                    )
    return sliding_weights
