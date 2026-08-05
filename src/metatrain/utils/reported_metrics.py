"""Extra per-target metrics, reported alongside RMSE and MAE.

A *metric* reports how the model is doing; a *loss* is what the gradients are taken of.
They are separate concerns that happen to share code, so any loss type can be reported
as a metric without being trained on.

One ``metrics`` block, at the top level of both ``options.yaml`` and ``eval.yaml``,
configures this for both mechanisms::

    metrics:
      mtt::rho_c_jfit:
        type: density_mse_via_c
        aux_basis: def2-universal-jfit
        subsets: [validation, test]   # optional; default: all of them
        log_interval: 5               # optional; default 1, training loop only

``mtt train`` uses it in the epoch loop (on the ``training`` and ``validation``
subsets) and again in the final evaluation; ``mtt eval`` uses it on whatever it is
given. RMSE and MAE are unaffected and keep being reported as before.
"""

from typing import Any, Dict, Optional

from .data import TargetInfo


#: Datasets a metric applies to unless it names its own.
DEFAULT_SUBSETS = ("training", "validation", "test")

#: Reported by every trainer already, through their own accumulators, and so not
#: configurable here. Asking for them by name would silently give a *second*, subtly
#: different number: the accumulators normalise per atom, whereas anything built here
#: is a loss averaged per system.
_BUILT_IN = {
    "rmse": "RMSE is always reported.",
    "mae": "set the 'log_mae' training hyperparameter to report MAE.",
}


def parse_metrics(
    config: Optional[Dict[str, Any]],
    targets: Dict[str, TargetInfo],
) -> Dict[str, Dict[str, Any]]:
    """
    Read the ``metrics`` block.

    :param config: The block, keyed by target name, or ``None``.
    :param targets: Every target of the run, used to reject unknown names.
    :return: One specification per target that configures a metric.
    :raises ValueError: If the block names a target the run does not have.
    """
    config = dict(config or {})
    unknown = set(config) - set(targets)
    if unknown:
        raise ValueError(
            f"the 'metrics' block configures {', '.join(sorted(unknown))}, which is "
            f"not among the targets of this run ({', '.join(sorted(targets))})."
        )

    parsed = {}
    for name, spec in config.items():
        if not isinstance(spec, dict):
            raise ValueError(
                f"the metric for target '{name}' must be a mapping with a 'type', "
                f"got {spec!r}."
            )
        if "type" not in spec:
            raise ValueError(f"the metric for target '{name}' is missing its 'type'.")
        metric_type = str(spec["type"]).lower()
        if metric_type in _BUILT_IN:
            raise ValueError(
                f"'{spec['type']}' cannot be requested in the 'metrics' block for "
                f"target '{name}': {_BUILT_IN[metric_type]} Configuring it here as "
                "well would report a second, differently normalised value under a "
                "near-identical name."
            )
        parsed[name] = dict(spec)
    return parsed


def metrics_for(
    specs: Dict[str, Dict[str, Any]],
    subset: Optional[str] = None,
    epochs_elapsed: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    The metrics that apply to one dataset right now.

    :param specs: Every configured metric, from :func:`parse_metrics`.
    :param subset: ``"training"``, ``"validation"`` or ``"test"``. ``None`` applies
        every metric, which is what a standalone ``mtt eval`` wants.
    :param epochs_elapsed: Epochs since the start of the run, for ``log_interval``.
        ``None`` in evaluation, which happens once.
    :return: The applicable specifications, stripped of the keys that select them.
    """
    selected = {}
    for name, spec in specs.items():
        if subset is not None and subset not in spec.get("subsets", DEFAULT_SUBSETS):
            continue
        if epochs_elapsed is not None:
            if epochs_elapsed % int(spec.get("log_interval", 1)) != 0:
                continue
        selected[name] = {
            key: value
            for key, value in spec.items()
            if key not in ("subsets", "log_interval")
        }
    return selected
