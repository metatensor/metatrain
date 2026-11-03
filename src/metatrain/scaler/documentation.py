r"""
Scaler
======

The scaler is a simple model that computes per-target and per-property scaling factors.
It is meant to be used as a preprocessing step for other architectures, so that targets
are standardized before being fed to the main model.

See :ref:`scale-targets` for more details.
"""

from typing import Dict, Optional, Union

from typing_extensions import NotRequired, TypedDict


FixedScalerWeights = Dict[str, Union[float, Dict[int, float]]]


class ModelHypers(TypedDict):
    """Hyperparameters for the scaler."""


class TrainerHypers(TypedDict):
    """Hyperparameters for the scaler trainer."""

    densify_atomic_basis: bool = True
    """Whether to atomic basis targets should be densified before being passed to the
    scaler.
    """
    fixed_weights: FixedScalerWeights = {}
    """Weights for target scaling.

    This is passed to the ``fixed_weights`` argument of
    :meth:`Scaler.train_model <metatrain.scaler.Scaler.train_model>`,
    see its documentation to understand exactly what to pass here.
    """
    additive_models: list[str] = []
    """List of checkpoint files to load additive models from.

    The contribution from these models will be subtracted from the targets
    before computing the scales.
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
    distributed: NotRequired[bool]
    """Whether to use distributed training. When not set, distributed training
    is enabled automatically when running under more than one SLURM task.
    Setting this option explicitly is deprecated."""
    distributed_port: int = 39591
    """Port for distributed communication among processes"""
    num_workers: Optional[int] = None
    """Number of workers for data loading. If not provided, it is set
    automatically."""
