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

Note that the uncertainties computed with this implementation are returned as
standard deviations, and not variances.
"""

from typing import Literal, Optional

from typing_extensions import TypedDict

from metatrain.utils.loss import LossSpecification


class ModelHypers(TypedDict):
    """Hyperparameters for the LLPR model."""

    num_ensemble_members: dict[str, int] = {}
    """Number of ensemble members for each target property for which LLPR ensembles
    should be constructed. No ensembles will be constructed for targets which are not
    listed.
    """


class TrainerHypers(TypedDict):
    """Hyperparameters for the LLPR trainer."""

    distributed: bool = False
    """Whether to use distributed training"""
    distributed_port: int = 39591
    """Port for distributed communication among processes"""
    batch_size: int = 8
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

    loss: str | dict[str, LossSpecification] = "gaussian_nll"
    """This section describes the loss function to be used during LLPR ensemble
    weight calibration. We strongly suggest only using ensemble-specific loss functions.
    Please refer to the :ref:`loss-functions` documentation for more details of the rest
    of the hypers."""

    num_epochs: Optional[int] = None
    """Number of epochs for which the LLPR ensemble weight calibration should
    take place. If set to ``null``, only the LLPR covariance matrix computation
    and calibration will be performed, without ensemble weight training."""

    train_all_parameters: bool = False
    """Whether to train all parameters of the LLPR-wrapped model, or only the
    ensemble weights. If ``true``, all parameters will be trained, including those
    of the base model. If ``false``, only the last-layer ensemble weights will be
    trained. Note that training all parameters (i.e., setting this flag to ``true``)
    will potentially change the uncertainty estimates given by the LLPR through the
    ``uncertainty`` outputs (because the last-layer features will change).
    In that case, only uncertainties calculated as standard deviations over the ensemble
    members (``ensemble`` outputs) will be meaningful."""

    warmup_fraction: float = 0.01
    """Fraction of training steps used for learning rate warmup."""

    learning_rate: float = 3e-4
    """Learning rate."""

    weight_decay: Optional[float] = None

    log_interval: int = 1
    """Interval to log metrics."""

    checkpoint_interval: int = 100
    """Interval to save checkpoints."""

    per_structure_targets: list[str] = []
    """Targets to calculate per-structure losses."""

    num_workers: Optional[int] = None
    """Number of workers for data loading. If not provided, it is set
    automatically."""

    log_mae: bool = False
    """Log MAE alongside RMSE"""

    log_separate_blocks: bool = False
    """Log per-block error."""

    best_model_metric: Literal["rmse_prod", "mae_prod", "loss"] = "loss"
    """Metric used to select best checkpoint (e.g., ``rmse_prod``)"""

    grad_clip_norm: float = 1.0
    """Maximum gradient norm value, by default inf (no clipping)"""

    calibration_method: Literal["crps", "nll"] = "nll"
    """Method used to calibrate the LLPR uncertainty via a multiplicative factor."""
