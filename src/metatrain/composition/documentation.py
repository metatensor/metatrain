r"""
Composition
===========

The composition model computes per-species contributions to invariant targets by solving
a deterministic least-squares problem. It is typically used as an additive baseline
within other architectures (e.g. PET, SOAP-BPNN, PhACE) to capture compositional offsets
before training the main model, but it can also be trained on its own. The main use case
for standalone training is to produce a checkpoint that initializes the composition
baseline of another architecture, avoiding a re-fit of the weights.

How the model works
-------------------

The composition model predicts a target as a sum of per-species weights :math:`w_t`, one
for each atomic type :math:`t`, fitted by least squares on the training data. Only
invariant quantities are fitted (energy being the typical example). For spherical
targets, these are blocks with :math:`\lambda = 0, \sigma = 1`. The fit can be bypassed
for chosen targets and atomic types with the ``atomic_baseline`` trainer hyperparameter,
which sets the corresponding weights to fixed, user-supplied values.

The exact form of the fit depends on the sample kind of the target.

For **per-structure** targets like energy (or anything with ``sample_kind`` set to
``"system"``), the prediction for a structure :math:`A` is

.. math::

    \hat{y}_A = \sum_{i \in A} w_{t_i} = \sum_t n^A_t \, w_t,

where :math:`t_i` is the type of atom :math:`i` and :math:`n^A_t` is the number of atoms
of type :math:`t` in :math:`A`. The weights minimize :math:`\sum_A \lVert y_A - \sum_t
n^A_t w_t \rVert^2`, i.e. they solve the normal equations :math:`X^\top X \, W = X^\top
Y`, where :math:`X_{At} = n^A_t`, :math:`Y` stacks the target values :math:`y_A`, and
:math:`W` stacks the weights :math:`w_t` (a small ridge term is added to the diagonal
for numerical stability).

For **per-atom** targets (i.e. anything with ``sample_kind`` set to ``"atom"``), the
prediction for atom :math:`i` is simply :math:`\hat{y}_i = w_{t_i}`. In this case
:math:`X` is a one-hot encoding of the atomic types, so :math:`X^\top X` is diagonal
(the per-type atom counts :math:`N_t`) and the fit reduces to

.. math::

    w_t = \frac{1}{N_t} \sum_{i \,:\, t_i = t} y_i,

that is, **the mean of the target over all atoms of type** :math:`t`. Types that never
appear in the training data are assigned a zero weight.

As a simple example, take a per-atom target where the hydrogen atoms carry the values
1.0 and 1.5 and a single oxygen atom carries 2.0: the fit gives :math:`w_\mathrm{H} =
1.25` and :math:`w_\mathrm{O} = 2.0`, so every hydrogen is predicted as 1.25 and every
oxygen as 2.0. For a per-structure energy, a water molecule would instead be predicted
as :math:`2 w_\mathrm{H} + w_\mathrm{O}`.

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

    distributed: NotRequired[bool] = False
    """Whether to use distributed training"""
    distributed_port: NotRequired[int] = 39591
    """Port for distributed communication among processes"""
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
    num_workers: NotRequired[Optional[int]] = None
    """Number of workers for data loading. If not provided, it is set
    automatically."""
