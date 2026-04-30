"""
E-PET (Experimental)
====================

`experimental.e_pet` integrates PET with tensor-basis spherical readouts while keeping
the PET trunk shared across all targets. Scalar targets use PET scalar heads. Spherical
targets are split internally into irrep blocks, and each block can either keep its own
PET head or share a PET head with selected irreps from the same target through
``irrep_head_groups``. Scalar targets and spherical irrep blocks can also share the
same PET head family across targets through ``shared_head_groups``.

Per-atom spherical atomic-basis targets follow PET's densified training path:
species-specific public blocks are densified internally, E-PET uses one target head
and one tensor basis per irrep, and evaluation sparsifies predictions back to the
public layout. True pair Hamiltonian targets are out of scope for this first pass.

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

from typing import Optional

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

    cutoff: SOAPCutoffConfig = init_with_defaults(SOAPCutoffConfig)
    """Radial cutoff configuration for the tensor-basis spherical expansion."""


class TensorBasisDefaults(TypedDict):
    """Default tensor-basis settings used for spherical targets."""

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


class ModelHypers(TypedDict):
    """Hyperparameters for the e-pet model."""

    pet: PETModelHypers = init_with_defaults(PETModelHypers)
    """Shared PET backbone hyperparameters."""

    tensor_basis_defaults: TensorBasisDefaults = init_with_defaults(
        TensorBasisDefaults
    )
    """Default tensor-basis settings for all spherical outputs."""

    volume_normalized_targets: list[str] = []
    """Targets that should be divided by structure volume after reconstruction."""

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

    Blocks omitted from the mapping keep private PET heads. Atomic-basis targets
    cannot be listed here in this first pass.
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


class TrainerHypers(PETTrainerHypers):
    """Hyperparameters for training e-pet models.

    The default split learning rates use the E-PET custom trainer path. This path
    does not currently support distributed training or finetuning.
    """

    learning_rate: float = 2.0e-4
    """Base learning rate used by scheduler defaults and fallback parameter groups."""

    coefficient_l2_weight: float = 0.0
    """Weight for the invariant coefficient L2 regularization."""

    basis_gram_weight: float = 0.0
    """Weight for the tensor-basis Gram penalty."""

    pet_trunk_learning_rate: Optional[float] = 2.0e-4
    """Learning rate for the shared PET backbone. Set to ``null`` to use
    ``learning_rate``."""

    tensor_basis_learning_rate: Optional[float] = 1.0e-3
    """Learning rate for tensor-basis modules. Set to ``null`` to use
    ``learning_rate``."""

    readout_learning_rate: Optional[float] = 1.0e-3
    """Learning rate for PET heads and final readout layers. Set to ``null`` to use
    ``learning_rate``."""
