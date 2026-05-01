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
two structurally identical heads (HOMO and LUMO) -- using PET's standard
per-target node + edge readout machinery -- produce per-atom scalar fields
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

Per-atom contributions follow the standard PET energy readout: a per-readout-
layer node MLP and edge MLP (each ``Linear -> SiLU -> Linear -> SiLU`` of width
``d_head``), followed by a per-readout-layer linear projection to a scalar, and
summed across all ``L`` GNN layers (with edge contributions weighted by the
PET cutoff factor):

.. math::

    h_i = \\sum_{l=1}^{L} \\Bigl[
        W^l_\\text{node} \\, \\phi(g_i^l) +
        \\sum_{j \\in \\mathcal{N}(i)} c_{ij} \\, W^l_\\text{edge} \\, \\phi(f_{ij}^l)
    \\Bigr],

where :math:`\\phi` is the two-layer MLP and :math:`W^l_\\bullet` is the
per-layer linear last layer. This is identical to the readout used for an
energy target in :ref:`PET <arch-pet>`; the only difference is the pooling
step. Heads use PET's ``model.d_head`` for the hidden width.

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

{{SECTION_DEFAULT_HYPERS}}
"""

from typing_extensions import TypedDict

from metatrain.pet.documentation import ModelHypers as PETModelHypers
from metatrain.pet.documentation import TrainerHypers as PETTrainerHypers
from metatrain.utils.hypers import init_with_defaults


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

    Inherits all the PET backbone hyperparameters and adds the GapPET-specific
    ``pooling`` section. The HOMO and LUMO heads share PET's ``d_head`` for the
    readout MLP width. The PET hyperparameter ``featurizer_type`` is forced to
    ``"residual"`` internally (all GNN layers are read out)."""

    pooling: PoolingHypers = init_with_defaults(PoolingHypers)
    """Hyperparameters for the extremal pooling."""


class TrainerHypers(PETTrainerHypers):
    """Hyperparameters for training a GapPET model.

    Identical to the PET trainer hyperparameters; reused as-is."""
