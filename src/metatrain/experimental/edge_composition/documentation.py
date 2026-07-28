from typing import Literal, Optional

from typing_extensions import TypedDict

from metatrain.utils.loss import LossSpecification


class ModelHypers(TypedDict):
    """Hyperparameters for the EdgeCompositionModel."""

    cutoff: float = 8.0

    sph_basis: Literal["coupled", "uncoupled"] = "coupled"
    """Whether to use coupled or uncoupled spherical harmonics basis."""

    radial_basis: Literal["exponential", "tabulated"] = "exponential"


class TrainerHypers(TypedDict):
    """Hyperparameters for the Trainer."""

    learning_rate: float = 1e-4
    """Learning rate of the optimizer."""

    distributed: bool = False
    """Whether to use distributed training"""
    distributed_port: int = 39591
    """Port for DDP communication"""
    batch_size: int = 16
    """The number of samples to use in each batch of training. This
    hyperparameter controls the tradeoff between training speed and memory usage. In
    general, larger batch sizes will lead to faster training, but might require more
    memory."""
    steps_per_batch: int = 10
    """Number of optimizer steps to take per batch."""
    num_epochs: int = 1000
    """Number of epochs."""

    log_interval: int = 1
    """Interval to log metrics."""
    checkpoint_interval: int = 100
    """Interval to save checkpoints."""
    scale_targets: bool = True
    """
    Normalize targets to unit std during training.

    If true, a single scale is computed for each target, given by the uncentered
    standard deviation across all values in the dataset for that target.

    For targets with more than one property (i.e. > 1 block or >= 1 block with > 1
    property), per-property scales are also computed, and used to re-scale model
    predictions.

    See also :ref:`scale-targets`.
    """
    num_workers: Optional[int] = None
    """Number of workers for data loading. If not provided, it is set automatically."""
    log_mae: bool = True
    """Log MAE alongside RMSE"""
    log_separate_blocks: bool = False
    """Log per-block error."""
    best_model_metric: Literal["rmse_prod", "mae_prod", "loss"] = "mae_prod"
    """Metric used to select best checkpoint (e.g., ``rmse_prod``)"""
    grad_clip_norm: float = 1.0
    """Maximum gradient norm value, by default inf (no clipping)"""
    loss: str | dict[str, LossSpecification] = "mse"
    """This section describes the loss function to be used. See the
    :ref:`loss-functions` for more details."""
    batch_atom_bounds: list[Optional[int]] = [None, None]
    """Bounds for the number of atoms per batch as [min, max]. Batches with atom
    counts outside these bounds will be skipped during training. Use ``None`` for
    either value to disable that bound. This is useful for preventing out-of-memory
    errors and ensuring consistent computational load. Default: ``[None, None]``."""

    missing_values: float = 0.0
    """Value to use for edges that are in the neighborlist but are not found
    in the target."""
