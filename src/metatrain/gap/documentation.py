"""
GAP
===

This is an implementation of the sparse Gaussian Approximation Potential
(GAP) :footcite:p:`bartok_representing_2013` using Smooth Overlap of Atomic Positions
(SOAP) :footcite:p:`bartok_gaussian_2010` implemented in `featomic <FEATOMIC_>`_.

.. _FEATOMIC: https://github.com/Luthaf/featomic

The GAP model in metatrain can only train on CPU, but evaluation
is also supported on GPU.

{SECTION_INSTALLATION}

{SECTION_DEFAULT_HYPERS}

{SECTION_MODEL_HYPERS}

with the following definitions needed to fully understand some of the parameters:

.. autoclass:: {architecture_path}.documentation.KRRHypers
    :members:
    :undoc-members:

.. autoclass:: {architecture_path}.documentation.SOAPHypers
    :members:
    :undoc-members:

.. autoclass:: {architecture_path}.documentation.SOAPCutoffHypers
    :members:
    :undoc-members:

.. autoclass:: {architecture_path}.documentation.SOAPCutoffSmoothingHypers
    :members:
    :undoc-members:

.. autoclass:: {architecture_path}.documentation.SOAPDensityHypers
    :members:
    :undoc-members:

.. autoclass:: {architecture_path}.documentation.SOAPDensityScalingHypers
    :members:
    :undoc-members:

.. autoclass:: {architecture_path}.documentation.SOAPBasisHypers
    :members:
    :undoc-members:

.. autoclass:: {architecture_path}.documentation.SOAPBasisRadialHypers
    :members:
    :undoc-members:

"""

from typing import Literal, Optional

from typing_extensions import TypedDict

from metatrain.utils.hypers import init_with_defaults


class SOAPCutoffSmoothingHypers(TypedDict):
    """Smoothing configuration for the SOAP descriptor cutoff."""

    type: Literal["ShiftedCosine"] = "ShiftedCosine"
    """Cutoff function used to smooth the behavior around the cutoff
    radius.

    Currently, only the Shifted Cosine function is supported, which is defined
    as ``f(r) = 1/2 * (1 + cos(π (r- cutoff + width) / width ))``."""
    width: float = 1.0


class SOAPCutoffHypers(TypedDict):
    """Cutoff configuration for the SOAP descriptor."""

    radius: float = 5.0
    """
    This should be set to a value after which most of the
    interactions between atoms is expected to be negligible.
    """

    smoothing: SOAPCutoffSmoothingHypers = init_with_defaults(SOAPCutoffSmoothingHypers)


class SOAPDensityScalingHypers(TypedDict):
    """Density scaling configuration for the SOAP descriptor."""

    type: Literal["Willatt2018"] = "Willatt2018"
    r"""Type of scaling, currently only one type is supported. It uses a
    long-range algebraic decay and smooth behavior at
    :math:`r\rightarrow 0`: as introduced by :footcite:t:`willatt_feature_2018` as
    ``f(r) = rate / (rate + (r / scale) ^ exponent)``."""

    rate: float = 1.0
    scale: float = 2.0
    exponent: float = 7.0


class SOAPDensityHypers(TypedDict):
    """Density configuration for the SOAP descriptor."""

    type: Literal["Gaussian"] = "Gaussian"
    """Currently, we only support a Gaussian type orbitals (GTO) as radial basis
    functions and radial integrals."""

    center_atom_weight: float = 1.0
    """Weight of the central atom contribution to the features. If
    1.0 the center atom contribution is weighted the same as any other
    contribution. If 0.0 the central atom does not contribute to the features
    at all."""

    width: float = 0.3
    """Width of the atom-centered gaussian creating the atomic
    density."""

    scaling: SOAPDensityScalingHypers = init_with_defaults(SOAPDensityScalingHypers)
    """Radial scaling can be used to reduce the importance of neighbor
    atoms further away from the center, usually improving the performance
    of the model.
    """


class SOAPBasisRadialHypers(TypedDict):
    """Radial basis configuration for the SOAP descriptor."""

    type: Literal["Gto"] = "Gto"

    max_radial: int = 7
    """Maximum radial channels of the spherical harmonics when
    computing the SOAP descriptors. In general, increasing this
    hyperparameter might lead to better accuracy, especially on larger
    datasets, at the cost of increased training and evaluation time."""


class SOAPBasisHypers(TypedDict):
    """Basis configuration for the SOAP descriptor."""

    type: Literal["TensorProduct"] = "TensorProduct"

    max_angular: int = 6
    """Maximum angular channels of the spherical harmonics when
    computing the SOAP descriptors. In general, increasing this
    hyperparameter might lead to better accuracy, especially on larger
    datasets, at the cost of increased training and evaluation time."""
    radial: SOAPBasisRadialHypers = init_with_defaults(SOAPBasisRadialHypers)


class SOAPHypers(TypedDict):
    """Configuration for the SOAP descriptors."""

    cutoff: SOAPCutoffHypers = init_with_defaults(SOAPCutoffHypers)
    """Spherical cutoff (Å) to use for atomic environments."""

    density: SOAPDensityHypers = init_with_defaults(SOAPDensityHypers)

    basis: SOAPBasisHypers = init_with_defaults(SOAPBasisHypers)


class KRRHypers(TypedDict):
    """Hyperparameters for the KRR model."""

    degree: Literal[2] = 2
    """Degree of the kernel. For now, only 2 is allowed."""

    num_sparse_points: int = 500
    """Number of sparse points to use during
    the training, it select the number of actual samples
    to use during the training. The selection is done with
    the Further Point Sampling (FPS) algorithm.
    The optimal number of sparse points depends on the system.
    Increasing it might impreve the accuracy, but it also increase the
    memory and time required for training."""


class ModelHypers(TypedDict):
    """Hyperparameters for the GAP model."""

    soap: SOAPHypers = init_with_defaults(SOAPHypers)

    krr: KRRHypers = init_with_defaults(KRRHypers)

    zbl: bool = False


class TrainerHypers(TypedDict):
    """Hyperparameters for the GAP trainer."""

    regularizer: float = 1e-3
    """Value of the regularizer for the energy. It should
    be tuned depending on the specific dataset. If it is too small might
    lead to overfitting, if it is too big might lead to bad accuracy."""

    regularizer_forces: Optional[float] = None
    """Value of the regularizer for the forces. It has
    a similar effect as ``regularizer``. If ``None``, it is set equal to
    ``regularizer``. It might be changed to have better accuracy."""
