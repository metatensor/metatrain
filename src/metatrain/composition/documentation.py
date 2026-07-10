"""
Composition
===========

The composition model computes per-species contributions to invariant
targets (e.g. energy) by solving a deterministic least-squares problem.
It is typically used as an additive baseline within other architectures
(e.g. PET, SOAP-BPNN, PhACE) to capture compositional offsets before
training the main model, but it can also be trained and used on its own.

{{SECTION_INSTALLATION}}

{{SECTION_DEFAULT_HYPERS}}

{{SECTION_MODEL_HYPERS}}

{{SECTION_TRAINER_HYPERS}}

{{SECTION_REFERENCES}}
"""

from typing import Dict, Optional, Union

from typing_extensions import NotRequired, TypedDict


FixedCompositionWeights = Dict[str, Union[float, Dict[int, float]]]


class ModelHypers(TypedDict):
    """Hyperparameters for the composition model."""

    pass


class TrainerHypers(TypedDict):
    """Hyperparameters for the composition trainer."""

    atomic_baseline: NotRequired[FixedCompositionWeights] = {}
    """Fixed per-species baselines, overriding the least-squares fit for the
    targets/atomic types they cover. A dict mapping each target name to either
    a single weight for all atomic types, or a dict mapping atomic types to
    weights. Unlike the identically-named hyperparameter of other
    architectures, a path to a composition checkpoint is not accepted here."""
    batch_size: NotRequired[Optional[int]] = None
    """Number of structures to accumulate at a time when building the
    least-squares problem. This only affects memory usage, not the fitted
    weights, since the composition model is a deterministic fit rather than
    an iterative optimization. Defaults to the size of the smallest training
    dataset."""
