"""
GapPET
======

GapPET predicts the electronic energy gap (e.g. HOMO-LUMO gap, or S0-S1 vertical
gap) of an atomic system in a **size-intensive** way, on top of the
:ref:`PET <arch-pet>` backbone. It is intended for training surrogates of
excited-state quantities used in non-adiabatic molecular dynamics: the model
predicts the gap and its gradient with respect to atomic positions, and is
typically composed externally with a separate ground-state MLIP to obtain the
excited-state potential energy surface (``E_S1 = E_S0 + E_gap``).

The architecture is identical to :ref:`PET <arch-pet>` except for the readout:
two structurally identical heads (HOMO and LUMO) produce per-atom scalar fields
``h_i^HOMO`` and ``h_i^LUMO``, which are pooled into per-system ``E_HOMO`` and
``E_LUMO`` via an extremal log-sum-exp:

.. math::

    E_\\text{HOMO} = \\frac{1}{\\alpha_H}
        \\log \\sum_i \\exp(\\alpha_H \\, h_i^\\text{HOMO}),
    \\qquad
    E_\\text{LUMO} = \\frac{1}{\\alpha_L}
        \\log \\sum_i \\exp(\\alpha_L \\, h_i^\\text{LUMO}),

with ``alpha_homo > 0`` (a smooth maximum, default ``+20``) and
``alpha_lumo < 0`` (a smooth minimum, default ``-20``). The predicted gap is
``E_gap = E_LUMO - E_HOMO``. Because the readout depends only on the *most
extreme* atomic contributions, the gap remains size-intensive: replicating the
simulation cell does not change the prediction.

Each head consists of two MLPs (one acting on per-atom node features, one on
per-edge features), shared across all GNN layers and summed across them:

.. math::

    h_i = \\sum_{l=1}^{L} \\Bigl[
        \\text{MLP}_\\text{node}(g_i^l) +
        \\sum_{j \\in \\mathcal{N}(i)} \\text{MLP}_\\text{edge}(f_{ij}^l)
    \\Bigr].

Forces are obtained by autograd through ``E_gap`` with respect to atomic
positions; per-atom HOMO/LUMO fields ``h_i^HOMO``, ``h_i^LUMO`` are exposed as
auxiliary outputs for interpretability (electron centroids, gyration radii,
etc.).

The PET ``residual`` featurizer is required (and is enforced internally), since
all ``L`` layer features are read out.

{{SECTION_INSTALLATION}}

Additional outputs
------------------

In addition to the gap target defined in the dataset, GapPET also outputs:

- ``mtt::aux::homo_per_atom``: the per-atom HOMO field ``h_i^HOMO``.
- ``mtt::aux::lumo_per_atom``: the per-atom LUMO field ``h_i^LUMO``.
- ``features``: the internal PET features (inherited from PET).
- :ref:`mtt-aux-target-last-layer-features`: the features for a given target
  (inherited from PET).

{{SECTION_DEFAULT_HYPERS}}
"""

from typing import Literal, Optional

from typing_extensions import TypedDict

from metatrain.pet.documentation import ModelHypers as PETModelHypers
from metatrain.pet.documentation import TrainerHypers as PETTrainerHypers
from metatrain.utils.hypers import init_with_defaults


class HeadHypers(TypedDict):
    """Hyperparameters for the HOMO/LUMO heads."""

    d_head: int = 128
    """Hidden width of the MLPs in each head. Both the node MLP and the edge MLP
    have one hidden layer of this width before projecting to a scalar."""

    d_head_homo: Optional[int] = None
    """Optional override for the HOMO head's ``d_head``. If ``None``, falls back
    to ``d_head``."""

    d_head_lumo: Optional[int] = None
    """Optional override for the LUMO head's ``d_head``. If ``None``, falls back
    to ``d_head``."""


class PoolingHypers(TypedDict):
    """Hyperparameters for the extremal (log-sum-exp) pooling."""

    alpha_homo: float = 20.0
    """Smoothness parameter for the HOMO pool. With ``alpha_homo > 0`` the pool
    is a smooth maximum; the limit ``alpha_homo -> +inf`` recovers ``max_i
    h_i^HOMO``."""

    alpha_lumo: float = -20.0
    """Smoothness parameter for the LUMO pool. With ``alpha_lumo < 0`` the pool
    is a smooth minimum; the limit ``alpha_lumo -> -inf`` recovers ``min_i
    h_i^LUMO``."""


class ModelHypers(PETModelHypers):
    """Hyperparameters for the GapPET model.

    Inherits all the PET backbone hyperparameters and adds two GapPET-specific
    sections, ``head`` and ``pooling``. The PET hyperparameter
    ``featurizer_type`` is forced to ``"residual"`` internally (all GNN layers
    are read out), and any user-supplied value is overridden with a warning.
    """

    head: HeadHypers = init_with_defaults(HeadHypers)
    """Hyperparameters for the HOMO and LUMO heads."""

    pooling: PoolingHypers = init_with_defaults(PoolingHypers)
    """Hyperparameters for the extremal pooling."""


class TrainerHypers(PETTrainerHypers):
    """Hyperparameters for training a GapPET model.

    Identical to the PET trainer hyperparameters; reused as-is."""
