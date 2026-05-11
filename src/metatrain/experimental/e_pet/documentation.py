"""
E-PET (Experimental)
====================

`experimental.e_pet` integrates PET with tensor-basis spherical readouts while keeping
the PET trunk shared across all targets. Scalar targets use PET scalar heads. Spherical
targets are split internally into irrep blocks, and each block can either keep its own
PET head or share a PET head with selected irreps from the same target through
``irrep_head_groups``. Scalar targets and spherical irrep blocks can also share the
same PET head family across targets through ``shared_head_groups``.

Cartesian rank-2 direct targets are accepted as public Cartesian tensors. E-PET
internally predicts hidden spherical ``l=0`` and ``l=2`` blocks, reconstructs the
Cartesian tensor, and applies loss, metrics, scaling, and optional volume
normalization to the public Cartesian target.

Per-atom spherical atomic-basis targets follow PET's densified training path:
species-specific public blocks are densified internally, E-PET uses one target head
by default and one tensor basis per irrep, and evaluation sparsifies predictions back
to the public layout. ``irrep_head_groups`` can opt selected atomic-basis irreps into
separate target-local heads without creating species/property-specific tensor bases.
True pair Hamiltonian targets are out of scope for this first pass.

{{SECTION_INSTALLATION}}

Additional outputs
------------------

- ``features``: the shared PET backbone features.
- ``mtt::features::{path}``: opt-in diagnostic captures from internal PET tensors.
- :ref:`mtt-aux-target-last-layer-features`: target-local head features before the
  final block-specific linear projections.

{{SECTION_DEFAULT_HYPERS}}

{{SECTION_MODEL_HYPERS}}

with the following definitions needed to fully understand some of the parameters:

.. autoclass:: {{architecture_path}}.documentation.TensorBasisDefaults
    :members:
    :undoc-members:

.. autoclass:: {{architecture_path}}.documentation.TensorBasisSOAPConfig
    :members:
    :undoc-members:

.. autoclass:: metatrain.soap_bpnn.documentation.SOAPCutoffConfig
    :members:
    :undoc-members:
"""

from typing import Dict, Literal, Optional

from typing_extensions import TypedDict

from metatrain.pet.documentation import (
    ModelHypers as PETModelHypers,
    TrainerHypers as PETTrainerHypers,
)
from metatrain.soap_bpnn.documentation import SOAPCutoffConfig
from metatrain.utils.hypers import init_with_defaults


class TensorBasisSOAPConfig(TypedDict):
    """SOAP-like settings read by the E-PET tensor-basis calculators."""

    max_radial: int = 7
    """Maximum number of radial channels used by the spherical expansion."""

    cutoff: SOAPCutoffConfig = {"radius": 4.5, "width": 0.5}
    """Radial cutoff configuration for the tensor-basis spherical expansion.

    The default radius matches the PET trunk cutoff default because E-PET evaluates
    tensor bases on the PET neighbor-list edges.
    """


class TensorBasisDefaults(TypedDict):
    """Default tensor-basis settings used for spherical and hidden spherical targets."""

    soap: TensorBasisSOAPConfig = init_with_defaults(TensorBasisSOAPConfig)
    """Spherical-expansion configuration for the tensor basis.

    The angular order is inferred from each target block's ``o3_lambda``. There is
    no separate ``max_angular`` / ``max_lambda`` setting for the lambda basis.
    """

    add_lambda_basis: bool = True
    """Whether to append a same-``lambda`` basis branch for tensorial targets."""

    extra_l1_vector_basis_branches: list[TensorBasisSOAPConfig] = [
        init_with_defaults(TensorBasisSOAPConfig)
    ]
    """Additional proper ``l=1`` vector-basis branches.

    The original ``VectorBasis`` is always present, so the default of one extra branch
    yields two ``VectorBasis`` branches in total.
    """

    legacy: bool = True
    """Whether to use the legacy tensor-basis species handling."""

    basis_normalization: Literal["none", "rms", "whiten"] = "none"
    """Default-off diagnostic normalization for tensor-basis blocks.

    ``"none"`` keeps the unnormalized path used by the current best reference runs.
    ``"rms"`` divides the whole local irrep basis by an invariant RMS scalar, which
    removes raw SOAP/spherical-harmonic/CG scale without changing equivariance.
    ``"whiten"`` uses a regularized inverse-square-root Gram transform to contract
    coefficients in a canonicalized local basis; it is intended as a diagnostic path.
    """

    basis_normalization_detach: bool = True
    """Whether the default-off RMS diagnostic should stop gradients through the
    RMS denominator."""

    basis_normalization_epsilon: float = 1.0e-6
    """Positive regularization used by default-off tensor-basis normalization
    diagnostics."""


class ModelHypers(TypedDict):
    """Hyperparameters for the e-pet model."""

    pet: PETModelHypers = init_with_defaults(PETModelHypers)
    """Shared PET backbone hyperparameters."""

    tensor_basis_defaults: TensorBasisDefaults = init_with_defaults(
        TensorBasisDefaults
    )
    """Default tensor-basis settings for all spherical outputs."""

    volume_normalized_targets: list[str] = []
    """Structure targets divided by cell volume after E-PET reconstruction.

    For Cartesian rank-2 stress targets, use the public Cartesian target name. Hidden
    spherical stress blocks are internal and should not be listed here.
    """

    irrep_head_groups: dict[str, dict[str, str]] = {}
    """Optional target-local mapping from ``\"<o3_lambda>,<o3_sigma>\"`` to a shared
    PET head identifier.

    Example:

    .. code-block:: yaml

        irrep_head_groups:
          quadrupole:
            "1,1": head_a
            "2,1": head_a
            "3,1": head_b

    Blocks omitted from the mapping keep private PET heads for ordinary spherical
    targets. For atomic-basis spherical targets, omitted blocks keep the default
    target-level PET head; configured blocks are split into the named target-local
    head groups.
    """

    shared_head_groups: dict[str, list[str]] = {}
    """Optional cross-target PET-head sharing groups.

    Each entry maps a user-defined group name to a list of selectors. Selectors are
    either scalar target names or explicit spherical irrep blocks in the form
    ``\"target[o3_lambda,o3_sigma]\"``.

    Example:

    .. code-block:: yaml

        shared_head_groups:
          stress_head:
            - mtt::stress_l0
            - mtt::stress_l2[2,1]

    Selectors in the same group share the PET ``node_heads`` / ``edge_heads`` while
    keeping separate final linear projections. Atomic-basis targets cannot be listed
    here in this first pass.
    """


class AtomicBasisIrrepBalancedLossHypers(TypedDict):
    """Default-off diagnostic E-PET loss for per-atom spherical atomic-basis targets.

    The target is first compared in physical sparse coefficient space. Blocks are
    then grouped by ``(o3_lambda, o3_sigma)``, normalized by one fitted RMS scale per
    group, and averaged equally over irreps. This prevents species/property count or
    very small per-property scales from dominating the optimization objective.
    Physical RMSE/MAE metrics and exported predictions are unchanged.
    """

    weight: float = 1.0
    """Overall weight multiplying the irrep-balanced target contribution."""

    scale: Literal["per_irrep_rms"] = "per_irrep_rms"
    """How to normalize each irrep group. ``"per_irrep_rms"`` uses one shared RMS
    of fitted scaler values for each ``(o3_lambda, o3_sigma)`` group."""


class TrainerHypers(PETTrainerHypers):
    """Hyperparameters for training e-pet models.

    The default split learning rates use the E-PET custom trainer path. This path
    does not currently support distributed training, finetuning, or PET's
    ``max_atoms_per_batch`` variable-size batching.
    """

    learning_rate: float = 2.0e-4
    """Base learning rate used by scheduler defaults and fallback parameter groups."""

    coefficient_l2_weight: float = 0.0
    """Weight for E-PET coefficient L2 regularization."""

    coefficient_l2_exclude_spherical_l0: bool = False
    """Whether to exclude spherical ``o3_lambda=0`` coefficient blocks from
    coefficient L2 regularization.

    This is useful when hidden or public scalar spherical blocks should remain close
    to PET-like scalar readouts while nontrivial tensor-basis blocks are regularized.
    """

    scale_property_floor_ratio: Optional[float] = None
    """Default-off diagnostic floor for fitted per-property scales. If set, each
    target's fitted scales are clamped to at least this ratio times that target's
    median positive scale."""

    atomic_basis_irrep_balanced_loss: Dict[str, AtomicBasisIrrepBalancedLossHypers] = {}
    """Default-off E-PET-only loss for listed per-atom spherical atomic-basis targets.

    Listed targets are excluded from the standard componentwise ``loss`` aggregation
    to avoid double-counting. If absent, E-PET uses the normal ``loss`` setting.
    """

    basis_gram_weight: float = 0.0
    """Weight for the E-PET tensor-basis Gram penalty."""

    pet_trunk_learning_rate: Optional[float] = 2.0e-4
    """Learning rate for the shared PET backbone. Set to ``null`` to use
    ``learning_rate``."""

    tensor_basis_learning_rate: Optional[float] = 1.0e-3
    """Learning rate for tensor-basis modules. Set to ``null`` to use
    ``learning_rate``."""

    readout_learning_rate: Optional[float] = 1.0e-3
    """Learning rate for PET heads and final readout layers. Set to ``null`` to use
    ``learning_rate``."""

    spherical_l0_readout_learning_rate: Optional[float] = None
    """Learning rate for spherical ``o3_lambda=0`` final readout layers. If an
    atomic-basis ``o3_lambda=0`` block is split into a dedicated target-local head with
    ``irrep_head_groups``, this also applies to that dedicated PET head. Set to
    ``null`` to use ``readout_learning_rate``."""
