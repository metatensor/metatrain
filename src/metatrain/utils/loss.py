# mypy: disable-error-code=misc
# We ignore misc errors in this file because TypedDict
# with default values is not allowed by mypy.
import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import metatensor.torch as mts
import torch
import torch.nn.functional as F
from metatensor.torch import Labels, TensorBlock, TensorMap
from pydantic import ConfigDict, with_config
from torch.nn.modules.loss import _Loss
from typing_extensions import NotRequired, TypedDict

from metatrain.utils.data import TargetInfo
from metatrain.utils.ensemble import uncertainty_output_name
from metatrain.utils.pyscf_loss import (
    METRICS,
    metric_matrix_name,
    ri_density_fit_constant_name,
    ri_projections_name,
    unpack_metric_matrices,
)


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

    #: Whether this loss must see its targets in the frame the dataset stores them in,
    #: rather than in the augmented frame. Set by losses that depend on a quantity
    #: derived from the unaugmented geometry; the trainer then augments only the
    #: systems and maps the predictions back before the loss is taken. See
    #: :func:`~metatrain.utils.augmentation.get_augmentation_transform`.
    evaluate_in_original_frame: bool = False

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


def _flatten_to_pyscf_order(
    tensor_map: TensorMap,
    subtract: Optional[TensorMap] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten atomic-basis coefficients (or their residual) into PySCF basis order.

    PySCF orders auxiliary basis functions by atom, then by shell in the order the
    basis lists them (angular momentum ascending, radial function within it), then
    by the ``2l + 1`` angular components. A densified atomic-basis target stores one
    block per ``o3_lambda``, with values ``(atom, m, n)`` and NaN on the property
    axis wherever an element has no such function.

    Laying the blocks out as ``(atom, [l][n][m])`` and taking the non-NaN entries in
    row-major order therefore reproduces PySCF order exactly, with no explicit index
    arithmetic: boolean-mask indexing walks rows in order, and the samples of a
    collated batch are already ordered by system and then by atom.

    :param tensor_map: Coefficients to flatten.
    :param subtract: If given, flatten ``tensor_map - subtract`` instead. NaN
        padding propagates through the subtraction and is dropped either way.
    :return: ``(flat, counts_per_atom)``, where ``flat`` concatenates every atom of
        the batch in order.
    """
    keys = sorted(tensor_map.keys, key=lambda key: int(key[0]))
    per_block: List[torch.Tensor] = []
    for key in keys:
        values = tensor_map.block(key).values
        if subtract is not None:
            values = values - subtract.block(key).values
        if int(key[0]) == 1:
            # metatensor orders l=1 components as m = -1, 0, +1; PySCF uses the
            # Cartesian-like x, y, z order, i.e. m = +1, -1, 0.
            values = values[:, [2, 0, 1], :]
        # (atom, m, n) -> (atom, n, m): n is major within an angular channel.
        per_block.append(values.transpose(1, 2).reshape(values.shape[0], -1))

    wide = torch.cat(per_block, dim=1)
    mask = ~torch.isnan(wide)
    return wide[mask], mask.sum(dim=1)


def _quadratic_form(vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """
    Evaluate ``v^T M v`` for one system.

    A matrix-vector product followed by a dot: the transformation this quadratic form
    actually is, rather than a general contraction. Both arguments reach here in the
    model's dtype, since ``batch_to`` casts the metric alongside the targets.

    :param vector: Coefficient (or residual) vector for one system.
    :param matrix: That system's two-centre metric matrix.
    :return: The scalar ``v^T M v``.
    """
    return torch.dot(vector, torch.mv(matrix, vector))


class _DensityLoss(LossInterface):
    """
    Shared machinery for the two quadratic density losses.

    Both express an error of the reconstructed scalar field as a quadratic form in
    the coefficients, weighted by a two-centre metric ``M`` supplied per system in
    ``extra_data``. They differ only in what reference data they consume; see
    :py:class:`DensityMSELossViaC` and :py:class:`DensityMSELossViaW`.

    :param name: key of the coefficient target.
    :param gradient: not supported; must be ``None``.
    :param weight: weight of this term in the aggregated loss.
    :param reduction: ``"mean"``, ``"sum"`` or ``"none"``.
    :param metric: ``"overlap"`` (S) or ``"coulomb"`` (J).
    :param aux_basis: auxiliary basis the reference coefficients were fitted in,
        e.g. ``"def2-universal-jfit"`` or ``"etb:def2-svp:2.0"``. Read by the trainer
        to build the metric transform, and kept here so the loss configuration is
        self-contained.
    """

    #: The metric matrix depends on the geometry, and is built on the unaugmented
    #: one -- the frame the reference coefficients were fitted in -- which is only
    #: consistent if the coefficients are compared in that same frame.
    evaluate_in_original_frame = True

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        reduction: str,
        metric: str = "overlap",
        aux_basis: Optional[str] = None,
    ):
        super().__init__(name, gradient, weight, reduction)
        if gradient is not None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support gradients of the coefficients."
            )
        if metric not in METRICS:
            raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}.")
        if aux_basis is None:
            raise ValueError(
                f"density losses on target '{name}' require 'aux_basis', the "
                "auxiliary basis the reference coefficients were fitted in."
            )
        self.metric = metric
        self.aux_basis = aux_basis

    def _require(self, extra_data: Optional[Any], key: str) -> Any:
        if extra_data is None or key not in extra_data:
            raise RuntimeError(
                f"'{type(self).__name__}' on target '{self.target}' requires "
                f"'{key}' in extra_data; it is added by the density collate "
                "transforms that the trainer installs."
            )
        return extra_data[key]

    def _per_system(
        self,
        tensor_map: TensorMap,
        subtract: Optional[TensorMap],
        extra_data: Optional[Any],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Flatten to PySCF order and split into one coefficient vector per system.

        Systems are kept ragged rather than padded to the batch maximum: the loss is
        a per-system quadratic form, so padding buys no batching that the metric
        matrices could share, while costing ``n_systems * n_max**2`` storage against
        the ``sum_i n_i**2`` actually needed.

        :param tensor_map: the predicted coefficients.
        :param subtract: reference coefficients to subtract, or ``None`` to
            flatten ``tensor_map`` alone.
        :param extra_data: the batch's extra data, holding the metric matrices.
        :return: ``(vectors, matrices)``, one of each per system.
        """
        packed = self._require(extra_data, metric_matrix_name(self.target, self.metric))
        matrices = unpack_metric_matrices(packed)

        system_of_atom = (
            tensor_map.block(tensor_map.keys[0]).samples.values[:, 0].to(torch.int64)
        )

        flat, counts_per_atom = _flatten_to_pyscf_order(tensor_map, subtract)

        counts = torch.zeros(
            len(matrices), dtype=counts_per_atom.dtype, device=counts_per_atom.device
        ).scatter_add_(0, system_of_atom, counts_per_atom)
        sizes = counts.tolist()

        expected = [matrix.shape[0] for matrix in matrices]
        if sizes != expected:
            raise ValueError(
                f"target '{self.target}' has a per-system coefficient count that "
                f"does not match the '{self.aux_basis}' auxiliary basis "
                f"({sizes} vs {expected}). Check that 'aux_basis' matches the basis "
                "the dataset was fitted in."
            )
        return list(torch.split(flat, sizes)), matrices

    def _reduce(self, per_system: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return per_system.mean()
        elif self.reduction == "sum":
            return per_system.sum()
        elif self.reduction == "none":
            return per_system
        raise ValueError(f"unknown reduction '{self.reduction}'")


class DensityMSELossViaC(_DensityLoss):
    """
    Quadratic density error from the coefficient residual: ``L = Δc^T M Δc``.

    For a field :math:`\\rho(r) = \\sum_i c_i \\phi_i(r)`, this is

    .. math::

        \\int |\\rho_\\mathrm{pred}(r) - \\rho_\\mathrm{ref}(r)|^2 \\, dr
            = \\Delta c^T S \\, \\Delta c

    with ``metric="overlap"``, and the electrostatic self-energy of the residual
    with ``metric="coulomb"``. Either is the quantity that matters for density
    learning, unlike a plain MSE on the coefficients, which ignores that the basis
    is neither orthogonal nor normalised.

    Requires reference coefficients in the dataset. Use
    :py:class:`DensityMSELossViaW` instead when the reference is stored as
    projections, or when ``S`` is too ill-conditioned to invert for them.

    The value is the **per-system total** error, an extensive quantity: metatrain's
    per-atom averaging does not apply, since it deliberately skips blocks whose
    samples carry an ``"atom"`` dimension, as this target's do.

    **Scale convention.** The trainer removes the per-target scale from the targets
    before the loss, so the residual seen here is :math:`\\Delta c / s`. Being
    quadratic, the loss is the true error divided by the constant :math:`s^2`, which
    is absorbed into ``weight``. A *per-property* scale would not factor out, but
    metatrain removes only the per-target scalar, so this is safe.
    """

    def compute(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Any] = None,
    ) -> torch.Tensor:
        deltas, matrices = self._per_system(
            predictions[self.target], targets[self.target], extra_data
        )
        # L_i = dc_i . (M_i dc_i): a matrix-vector product followed by a dot, which
        # is the transformation this quadratic form actually is. The contraction is
        # bandwidth-bound on streaming M, so the arrangement matters less than not
        # streaming padding along with it.
        return self._reduce(
            torch.stack(
                [
                    _quadratic_form(delta, matrix)
                    for delta, matrix in zip(deltas, matrices, strict=True)
                ]
            )
        )


class DensityMSELossViaW(_DensityLoss):
    """
    Quadratic density error from projections: ``L = c^T M c - 2 c^T w (+ const)``.

    Here :math:`w = M c_\\mathrm{ref}` are the projections of the reference density
    onto the auxiliary basis. Expanding :math:`\\Delta c^T M \\Delta c` gives

    .. math::

        c^T M c - 2 c^T w + c_\\mathrm{ref}^T w

    so this is the *same* quantity as :py:class:`DensityMSELossViaC`, reached without
    ever forming :math:`c_\\mathrm{ref} = M^{-1} w`. That matters because ``S`` is
    typically ill-conditioned for large auxiliary bases, so the inversion — not the
    loss — is where accuracy is lost. The two therefore make a meaningful ablation
    pair rather than a redundant one.

    The final term is a constant with no gradient. It is added when the collate
    transform has supplied it, which makes the loss bounded below by zero and hence
    directly comparable to ``via_c``; without it the loss is shifted by an unknown
    per-system constant and only differences are meaningful.

    :param name: key of the coefficient target.
    :param gradient: not supported; must be ``None``.
    :param weight: weight of this term in the aggregated loss.
    :param reduction: ``"mean"``, ``"sum"`` or ``"none"``.
    :param metric: ``"overlap"`` (S) or ``"coulomb"`` (J).
    :param aux_basis: auxiliary basis the reference coefficients were fitted in.
    :param projections_key: ``extra_data`` key holding ``w``. Defaults to
        ``<target>_projections``.
    """

    def __init__(
        self,
        name: str,
        gradient: Optional[str],
        weight: float,
        reduction: str,
        metric: str = "overlap",
        aux_basis: Optional[str] = None,
        projections_key: Optional[str] = None,
    ):
        super().__init__(name, gradient, weight, reduction, metric, aux_basis)
        self.projections_key = (
            projections_key
            if projections_key is not None
            else ri_projections_name(name)
        )

    def compute(
        self,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Any] = None,
    ) -> torch.Tensor:
        projections = self._require(extra_data, self.projections_key)

        # Flatten predictions and projections under the same code path so the two
        # flat vectors index the same basis functions.
        coefficients, matrices = self._per_system(
            predictions[self.target], None, extra_data
        )
        projected, _ = self._per_system(projections, None, extra_data)

        per_system = torch.stack(
            [
                _quadratic_form(c, matrix)
                - 2.0 * torch.dot(c.to(matrix.dtype), w.to(matrix.dtype))
                for c, w, matrix in zip(coefficients, projected, matrices, strict=True)
            ]
        )

        constant_name = ri_density_fit_constant_name(self.target)
        if extra_data is not None and constant_name in extra_data:
            constant = extra_data[constant_name].block().values.reshape(-1)
            per_system = per_system + constant.to(
                dtype=per_system.dtype, device=per_system.device
            )
        return self._reduce(per_system)


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
    DENSITY_MSE_VIA_C = ("density_mse_via_c", DensityMSELossViaC)
    DENSITY_MSE_VIA_W = ("density_mse_via_w", DensityMSELossViaW)

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


def build_reported_losses(
    specs: Dict[str, Any],
    targets: Dict[str, TargetInfo],
) -> Dict[str, "LossAggregator"]:
    """
    Build losses that are evaluated and reported, but never trained on.

    One aggregator is built per target rather than one for all of them, so that each
    is reported under its own name.

    :param specs: Loss specifications keyed by target name.
    :param targets: Target information, used to build the losses.
    :return: Mapping from target name to the aggregator reporting on it.
    """
    aggregators = {}
    for target_name, spec in specs.items():
        # These specifications are not reached by the hypers machinery that fills in
        # defaults for the top-level ones, so complete them here.
        complete = LossSpecification(
            {"type": "mse", "weight": 1.0, "reduction": "mean", "gradients": {}}
        )
        complete.update(spec)
        aggregators[target_name] = LossAggregator(
            targets={target_name: targets[target_name]},
            config={target_name: complete},
        )
    return aggregators


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
