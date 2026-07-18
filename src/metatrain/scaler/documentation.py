r"""
Scaler
======

The scaler is a simple model that computes per-target and per-property scaling factors.
It is meant to be used as a preprocessing step for other architectures, so that targets
are standardized before being fed to the main model.

See :ref:`the scaler documentation <scaler>` for more details.
"""

from typing import Dict, Optional, Union

from typing_extensions import NotRequired, TypedDict


FixedScalerWeights = Dict[str, Union[float, Dict[int, float]]]


class ModelHypers(TypedDict):
    """Hyperparameters for the scaler."""

    densify_atomic_basis: NotRequired[bool] = True
    """Whether to densify the atomic basis targets when computing the scaling
    weights. This can only be done if the target is loaded from a DiskDataset.

    Most models will require the scaler to work with the densified atomic basis.
    """


class TrainerHypers(TypedDict):
    """Hyperparameters for the scaler trainer."""

    fixed_weights: FixedScalerWeights
    """Weights for target scaling.

    This is passed to the ``fixed_weights`` argument of
    :meth:`Scaler.train_model <metatrain.scaler.Scaler.train_model>`,
    see its documentation to understand exactly what to pass here.
    """
    batch_size: Optional[int] = None
    """Number of structures to accumulate at a time.
    This only affects memory usage, not the resulting scales, since the
    scaler is a deterministic modelrather than an iterative optimization.
    Defaults to the size of the smallest training dataset.
    """
    per_structure_targets: list[str] = []
    """Target names that should be treated as
    per-structure quantities and therefore not divided by the number of atoms.
    """
