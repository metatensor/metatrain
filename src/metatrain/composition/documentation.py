"""
Composition
===========

The composition model computes per-species contributions to invariant
targets (e.g. energy) by solving a deterministic least-squares problem.
It is typically used as an additive baseline within other architectures
(e.g. PET, SOAP-BPNN, PhACE) to capture compositional offsets before
training the main model.
"""

from typing import Dict, Union

from typing_extensions import NotRequired, TypedDict


FixedCompositionWeights = Dict[str, Union[float, Dict[int, float]]]


class ModelHypers(TypedDict):
    """Hyperparameters for the composition model."""

    pass


class TrainerHypers(TypedDict):
    """Hyperparameters for the composition trainer."""

    atomic_baseline: NotRequired[FixedCompositionWeights]
    batch_size: NotRequired[int]
