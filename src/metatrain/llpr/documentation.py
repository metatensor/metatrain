"""
LLPR
====

.. image:: https://codecov.io/gh/metatensor/metatrain/branch/main/graph/badge.svg?flag=coverage_llpr
   :target: https://codecov.io/gh/metatensor/metatrain/tree/main/src/metatrain/llpr

The LLPR architecture is a "wrapper" architecture that enables cheap uncertainty
quantification (UQ) via the last-layer prediction rigidity (LLPR) approach proposed
by Bigi et al. :footcite:p:`bigi_mlst_2024` It is compatible with the following
``metatrain`` models constructed from NN-based architectures: PET and SOAP-BPNN.
The implementation of the LLPR as a separate architecture within ``metatrain``
allows the users to compute the uncertainties without dealing with the fine details
of the LLPR implementation.

{{SECTION_INSTALLATION}}

{{SECTION_DEFAULT_HYPERS}}

{{SECTION_MODEL_HYPERS}}

where the ensemble hyperparameters should adhere to the following structure:

.. autoclass:: {{architecture_path}}.documentation.EnsemblesHypers
    :members:
    :undoc-members:

"""

from typing import Optional

from typing_extensions import TypedDict

from metatrain.utils.hypers import init_with_defaults


class EnsemblesHypers(TypedDict):
    """Configuration of ensembles in LLPR."""

    means: dict[str, list[str]] = {}
    """This accepts a dictionary of targets and the names of their corresponding
    last-layer weights. For example, in the case of energy trained with the default
    ``energy`` key in a PET model, the following could be the set of weights to provide:

    .. code-block:: yaml

      means:
        energy:
          - node_last_layers.energy.0.energy___0.weight
          - node_last_layers.energy.1.energy___0.weight
          - edge_last_layers.energy.0.energy___0.weight
          - edge_last_layers.energy.1.energy___0.weight
    """

    num_members: dict[str, int] = {}
    """This is a dictionary of targets and the corresponding number of ensemble
    members to sample. Note that a sufficiently large number of members (more than 16)
    are required for robust uncertainty propagation.
    (e.g. ``num_members: {energy: 128}``)
    """


class ModelHypers(TypedDict):
    """Hyperparameters for the LLPR model."""

    ensembles: EnsemblesHypers = init_with_defaults(EnsemblesHypers)
    """To perform uncertainty propagation, one can generate an ensemble of weights
    from the calibrated inverse covariance matrix from the LLPR formalism.
    """


class TrainerHypers(TypedDict):
    """Hyperparameters for the LLPR trainer."""

    batch_size: int = 12
    """This defines the batch size used in the computation of last-layer
    features, covariance matrix, etc."""

    regularizer: Optional[float] = None
    r"""This is the regularizer value :math:`\varsigma` that is used in
    applying Eq. 24 of Bigi et al :footcite:p:`bigi_mlst_2024`:

    .. math::

        \sigma^2_\star = \alpha^2 \boldsymbol{\mathrm{f}}^{\mathrm{T}}_\star
        (\boldsymbol{\mathrm{F}}^{\mathrm{T}} \boldsymbol{\mathrm{F}} + \varsigma^2
        \boldsymbol{\mathrm{I}})^{-1} \boldsymbol{\mathrm{f}}_\star

    If set to ``null``, the internal routine will determine the smallest regularizer
    value that guarantees numerical stability in matrix inversion. Having exposed the
    formula here, we also note to the user that the training routine of the LLPR
    wrapper model finds the ideal global calibration factor :math:`\alpha`."""

    model_checkpoint: Optional[str] = None
    """This should provide the checkpoint to the model for which the
    user wants to perform UQ based on the LLPR approach. Note that the model
    architecture must comply with the requirement that the last-layer features are
    exposed under the convention defined by metatrain."""
