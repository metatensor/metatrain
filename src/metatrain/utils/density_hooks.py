"""Trainer-side support for density (RI-coefficient) losses.

A density loss needs one thing from a trainer that a pointwise loss does not: a
two-centre metric matrix attached to every batch, built from the system's geometry.
This module collects that behind one object, so that adding density support to a
trainer is a single splice into its collate functions.

A trainer wires it in like this::

    density = get_density_hooks(self.hypers["loss"])

    CollateFn(
        target_keys,
        callables=[
            atomic_basis_transform,
            *density.training_collate_transforms(),  # before augmentation
            augmentation_callable,
            *base_callables,
        ],
    )
    CollateFn(
        target_keys,
        callables=[
            atomic_basis_transform,
            *density.validation_collate_transforms(),
            *base_callables,
        ],
    )

Without a density loss both lists are empty, so the trainer needs no conditionals and
pays nothing.

The two lists differ because a density loss may be configured as a *metric* rather
than trained on. A metric is evaluated on validation only, and building its matrices
is expensive, so the training collate must not build them.

The one ordering constraint is that these run **before** augmentation: the metric
depends on the geometry, and it is the *unaugmented* geometry the reference
coefficients were fitted in. Comparing coefficients in that same frame is then
handled generically -- the losses declare
:attr:`~metatrain.utils.loss.LossInterface.evaluate_in_original_frame`, and
:func:`~metatrain.utils.augmentation.get_augmentation_transform` picks the
augmentation workflow that honours it. None of that is density-specific, and none of
it lives here.
"""

from typing import Any, Callable, Dict, List, Optional, Union

from .pyscf_loss import get_metric_matrices_transform


#: Loss types that need auxiliary-basis metric matrices attached to each batch.
DENSITY_LOSS_TYPES = ("density_mse_via_c", "density_mse_via_w")


def _metric_transforms(
    aux_bases_by_metric: Dict[str, Dict[str, str]],
) -> List[Callable]:
    """One transform per metric; targets sharing a basis share one computation.

    :param aux_bases_by_metric: ``{metric: {target: aux_basis}}``.
    :return: The collate transforms.
    """
    return [
        get_metric_matrices_transform(targets_map, metric)
        for metric, targets_map in aux_bases_by_metric.items()
    ]


class DensityLossHooks:
    """
    Everything a trainer must do to support density losses.

    Build with :func:`get_density_hooks` rather than directly; it returns an inactive
    instance when no density loss is configured.

    :param trained: Mapping from metric name to the ``{target: aux_basis}`` served by
        it, for losses that are trained on.
    :param reported: The same, for losses that are only reported as metrics.
    """

    def __init__(
        self,
        trained: Dict[str, Dict[str, str]],
        reported: Dict[str, Dict[str, str]],
    ) -> None:
        self._trained = trained
        self._reported = reported

    def training_collate_transforms(self) -> List[Callable]:
        """
        Metric-matrix transforms for training batches, to run **before** augmentation.

        :return: Collate transforms; empty when nothing is trained on a density loss.
        """
        return _metric_transforms(self._trained)

    def validation_collate_transforms(self) -> List[Callable]:
        """
        Metric-matrix transforms for validation batches.

        Covers both the trained losses -- which are also evaluated on validation --
        and any reported as metrics.

        :return: Collate transforms; empty when inactive.
        """
        combined: Dict[str, Dict[str, str]] = {
            metric: dict(targets_map) for metric, targets_map in self._trained.items()
        }
        for metric, targets_map in self._reported.items():
            combined.setdefault(metric, {}).update(targets_map)
        return _metric_transforms(combined)


def _aux_bases_by_metric(specs: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Group the density losses among ``specs`` by the metric they need.

    :param specs: Loss specifications keyed by target name.
    :return: ``{metric: {target: aux_basis}}``, empty when none is a density loss.
    """
    grouped: Dict[str, Dict[str, str]] = {}
    for target_name, spec in specs.items():
        if not isinstance(spec, dict) or spec.get("type") not in DENSITY_LOSS_TYPES:
            continue
        metric = spec.get("metric", "overlap")
        grouped.setdefault(metric, {})[target_name] = spec["aux_basis"]
    return grouped


def get_density_hooks(
    loss_hypers: Union[str, Dict[str, Any], None],
    metrics: Optional[Dict[str, Any]] = None,
) -> DensityLossHooks:
    """
    Build the density hooks a trainer needs, or an inactive instance if none.

    :param loss_hypers: The trainer's ``loss`` hyperparameter, keyed by target name.
        A string (the shorthand for "this loss type for every target") configures no
        density loss.
    :param metrics: The ``metrics`` block, keyed by target name. A density metric is
        evaluated on validation only, so its matrices are not built for training
        batches.
    :return: The hooks for this configuration.
    """
    trained = _aux_bases_by_metric(loss_hypers) if isinstance(loss_hypers, dict) else {}
    return DensityLossHooks(trained, _aux_bases_by_metric(metrics or {}))
