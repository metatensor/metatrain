"""Trainer-side support for density (RI-coefficient) losses.

A density loss needs one thing from a trainer that a pointwise loss does not: a
two-centre metric matrix attached to every batch, built from the system's geometry.
This module collects that behind one object, so that adding density support to a
trainer is a single splice into its collate function.

A trainer wires it in like this::

    density = get_density_hooks(self.hypers["loss"])

    CollateFn(
        target_keys,
        callables=[
            atomic_basis_transform,
            *density.collate_transforms(),  # before augmentation
            augmentation_callable,
            *base_callables,
        ],
    )

Without a density loss the list is empty, so the trainer needs no conditionals and
pays nothing.

The one ordering constraint is that these run **before** augmentation: the metric
depends on the geometry, and it is the *unaugmented* geometry the reference
coefficients were fitted in. Comparing coefficients in that same frame is then
handled generically -- the losses declare
:attr:`~metatrain.utils.loss.LossInterface.evaluate_in_original_frame`, and
:func:`~metatrain.utils.augmentation.get_augmentation_transform` picks the
augmentation workflow that honours it. None of that is density-specific, and none of
it lives here.
"""

from typing import Any, Callable, Dict, List, Union

from .pyscf_loss import get_metric_matrices_transform


#: Loss types that need auxiliary-basis metric matrices attached to each batch.
DENSITY_LOSS_TYPES = ("density_mse_via_c", "density_mse_via_w")


class DensityLossHooks:
    """
    Everything a trainer must do to support density losses.

    Build with :func:`get_density_hooks` rather than directly; it returns an inactive
    instance when no density loss is configured.

    :param aux_bases_by_metric: Mapping from metric name to the
        ``{target: aux_basis}`` served by it.
    """

    def __init__(self, aux_bases_by_metric: Dict[str, Dict[str, str]]) -> None:
        self._aux_bases_by_metric = aux_bases_by_metric

    def collate_transforms(self) -> List[Callable]:
        """
        The metric-matrix transforms, to run **before** augmentation.

        One transform per distinct metric.

        :return: Collate transforms; empty when inactive.
        """
        return [
            get_metric_matrices_transform(targets_map, metric)
            for metric, targets_map in self._aux_bases_by_metric.items()
        ]


def get_density_hooks(
    loss_hypers: Union[str, Dict[str, Any], None],
) -> DensityLossHooks:
    """
    Build the density hooks a trainer needs, or an inactive instance if none.

    :param loss_hypers: The trainer's ``loss`` hyperparameter, keyed by target name.
        A string (the shorthand for "this loss type for every target") configures no
        density loss.
    :return: The hooks for this configuration.
    """
    # Group by metric: each needs its own set of matrices, and targets sharing one
    # share the computation.
    aux_bases_by_metric: Dict[str, Dict[str, str]] = {}
    if isinstance(loss_hypers, dict):
        for target_name, spec in loss_hypers.items():
            if not isinstance(spec, dict) or spec.get("type") not in DENSITY_LOSS_TYPES:
                continue
            metric = spec.get("metric", "overlap")
            aux_bases_by_metric.setdefault(metric, {})[target_name] = spec["aux_basis"]

    return DensityLossHooks(aux_bases_by_metric)
