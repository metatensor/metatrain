"""
LLPR
====

The LLPR architecture is a "wrapper" architecture that enables cheap uncertainty
quantification (UQ) via the last-layer prediction rigidity (LLPR) approach proposed
by Bigi et al. :footcite:p:`bigi_mlst_2024` It is compatible with the following
``metatrain`` models constructed from NN-based architectures: PET and SOAP-BPNN.
The implementation of the LLPR as a separate architecture within ``metatrain``
allows the users to compute the uncertainties without dealing with the fine details
of the LLPR implementation.

This implementation further allows the user to perform gradient-based tuning of
the ensemble weights sampled from the LLPR formalism, which can lead to improved
uncertainty estimates. Gradients (e.g. forces and stresses) are not yet used in this
implementation of the LLPR.

{{SECTION_INSTALLATION}}

{{SECTION_DEFAULT_HYPERS}}

{{SECTION_MODEL_HYPERS}}

where the ensemble hyperparameters should adhere to the following structure:

.. autoclass:: {{architecture_path}}.documentation.EnsemblesHypers
    :members:
    :undoc-members:

"""

from typing import Literal, Optional

from typing_extensions import TypedDict

from metatrain.utils.additive import FixedCompositionWeights
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification
from metatrain.utils.scaler import FixedScalerWeights


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

    mode: Literal["llpr_only", "llpr_ens_calib"] = "llpr_only"
    """This determines the mode of the LLPR model training. Available options are:
    "llpr_only", which only computes the covariance matrix and calibrates the
    multipliers for each target, and "llpr_ens_calib", which then samples an ensemble
    and performs gradient-based training to further calibrate the ensemble weights."""

    model_checkpoint: Optional[str] = None
    """This should provide the checkpoint to the model for which the
    user wants to perform UQ based on the LLPR approach. Note that the model
    architecture must comply with the requirement that the last-layer features are
    exposed under the convention defined by metatrain."""

    ens_calib_loss: str | dict[str, LossSpecification] = "ensemble_nll"
    """This section describes the loss function to be used during LLPR ensemble
    weight calibration. We strongly suggest only using "ensemble_nll" loss. see
    :ref:`loss-functions` for more details of the rest of the hypers."""

    calib_options: dict = {}
    """TODO: MISSING DOCS!!"""

    num_epochs: Optional[int] = 0
    """Number of epochs for which the LLPR ensemble weight calibration should 
    take place. Ignored when mode is set to "llpr_only"."""

    warmup_fraction: float = 0.01
    """Fraction of training steps used for learning rate warmup."""

    learning_rate: float = 1e-4
    """Learning rate."""

    weight_decay: Optional[float] = None

    log_interval: int = 1
    """Interval to log metrics."""

    checkpoint_interval: int = 100
    """Interval to save checkpoints."""

    scale_targets: bool = True
    """Normalize targets to unit std during training."""

    fixed_composition_weights: FixedCompositionWeights = {}
    """Weights for atomic contributions.

    This is passed to the ``fixed_weights`` argument of
    :meth:`CompositionModel.train_model
    <metatrain.utils.additive.composition.CompositionModel.train_model>`,
    see its documentation to understand exactly what to pass here.
    """

    remove_composition_contribution: bool = True
    """Whether to remove the atomic composition contribution from the
    targets by fitting a linear model to the training data before
    training the neural network."""

    fixed_scaling_weights: FixedScalerWeights = {}
    """Weights for target scaling.

    This is passed to the ``fixed_weights`` argument of
    :meth:`Scaler.train_model <metatrain.utils.scaler.scaler.Scaler.train_model>`,
    see its documentation to understand exactly what to pass here.
    """

    per_structure_targets: list[str] = []
    """Targets to calculate per-structure losses."""

    num_workers: Optional[int] = None
    """Number of workers for data loading. If not provided, it is set
    automatically."""

    log_mae: bool = True
    """Log MAE alongside RMSE"""

    log_separate_blocks: bool = False
    """Log per-block error."""

    best_model_metric: Literal["rmse_prod", "mae_prod", "loss"] = "mae_prod"
    """Metric used to select best checkpoint (e.g., ``rmse_prod``)"""

    grad_clip_norm: float = 1.0
    """Maximum gradient norm value, by default inf (no clipping)"""
