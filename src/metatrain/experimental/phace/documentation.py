"""
PhACE
=====

PhACE is a cleaner, more user-friendly reimplementation of the original
PET model :footcite:p:`pozdnyakov_smooth_2023`. It is designed for better
modularity and maintainability, while preseving compatibility with the original
PET implementation in ``metatrain``. It also adds new features like long-range
features, better fine-tuning implementation, a possibility to train on
arbitrarty targets, and a faster inference due to the ``fast attention``.

{{SECTION_INSTALLATION}}

{{SECTION_DEFAULT_HYPERS}}

Tuning hyperparameters
----------------------

The default hyperparameters above will work well in most cases, but they
may not be optimal for your specific dataset. There is good number of
parameters to tune, both for the
:ref:`model <architecture-{{architecture}}_model_hypers>` and the
:ref:`trainer <architecture-{{architecture}}_trainer_hypers>`. Since seeing them
for the first time might be overwhelming, here we provide a **list of the
parameters that are in general the most important** (in decreasing order
of importance):

"""

from typing import Literal, Optional

from typing_extensions import TypedDict

from metatrain.utils.additive import FixedCompositionWeights
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.long_range import LongRangeHypers
from metatrain.utils.loss import LossSpecification
from metatrain.utils.scaler import FixedScalerWeights


class RadialBasisHypers(TypedDict):
    """In some systems and datasets, enabling long-range Coulomb interactions
    might be beneficial for the accuracy of the model and/or
    its physical correctness."""

    max_eigenvalue: float = 25.0
    """Maximum eigenvalue for the radial basis."""

    scale: float = 0.7
    """Scaling factor for the radial basis."""

    optimizable_lengthscales: bool = False
    """Whether the length scales in the radial basis are optimizable."""


###########################
#  MODEL HYPERPARAMETERS  #
###########################


class ModelHypers(TypedDict):
    """Hyperparameters for the experimental.phace model."""

    max_correlation_order_per_layer: int = 3
    """Maximum correlation order per layer."""

    num_message_passing_layers: int = 2
    """Number of message passing layers."""

    cutoff: float = 5.0
    """Cutoff radius for neighbor search."""

    cutoff_width: float = 1.0
    """Width of the cutoff smoothing function."""

    num_element_channels: int = 128
    """Number of channels per element."""

    force_rectangular: bool = False
    """Makes the number of channels per irrep the same."""

    spherical_linear_layers: bool = False
    """Whether to perform linear layers in the spherical representation"""

    radial_basis: RadialBasisHypers = init_with_defaults(RadialBasisHypers)
    """Hyperparameters for the radial basis functions."""

    nu_scaling: float = 0.1
    """Scaling for the nu term."""

    mp_scaling: float = 0.1
    """Scaling for message passing."""

    overall_scaling: float = 1.0
    """Overall scaling factor."""

    disable_nu_0: bool = True
    """Whether to disable nu=0."""

    use_sphericart: bool = False
    """Whether to use spherical Cartesian coordinates."""

    head_num_layers: int = 1
    """Number of layers in the head."""

    heads: dict[str, Literal["linear", "mlp"]] = {}
    """Heads to use in the model, with options being "linear" or "mlp"."""

    zbl: bool = False
    """Whether to use the ZBL potential in the model."""

    long_range: LongRangeHypers = init_with_defaults(LongRangeHypers)
    """Long-range Coulomb interactions parameters."""


##############################
#  TRAINER HYPERPARAMETERS   #
##############################


class TrainerHypers(TypedDict):
    """Hyperparameters for training the experimental.phace model."""

    compile: bool = True
    """Whether to use `torch.compile` during training."""

    distributed: bool = False
    """Whether to use distributed training."""

    distributed_port: int = 39591
    """Port for DDP communication."""

    batch_size: int = 8
    """Batch size for training."""

    num_epochs: int = 1000
    """Number of epochs to train the model."""

    learning_rate: float = 0.01
    """Learning rate for the optimizer."""

    warmup_fraction: float = 0.01
    """Fraction of training steps for learning rate warmup."""

    gradient_clipping: Optional[float] = None
    """Gradient clipping value. If None, no clipping is applied."""

    scheduler_patience: int = 100
    """Patience for the learning rate scheduler."""

    scheduler_factor: float = 0.7
    """Factor by which to reduce the learning rate in the scheduler."""

    log_interval: int = 1
    """Interval to log metrics during training."""

    checkpoint_interval: int = 25
    """Interval to save model checkpoints."""

    scale_targets: bool = True
    """Whether to scale targets during training."""

    fixed_composition_weights: FixedCompositionWeights = {}
    """Fixed weights for atomic composition during training."""

    fixed_scaling_weights: FixedScalerWeights = {}
    """Fixed scaling weights for the model."""

    num_workers: Optional[int] = None
    """Number of workers for data loading."""

    per_structure_targets: list[str] = []
    """List of targets to calculate per-structure losses."""

    log_separate_blocks: bool = False
    """Whether to log per-block error during training."""

    log_mae: bool = False
    """Whether to log MAE alongside RMSE during training."""

    best_model_metric: Literal["rmse_prod", "mae_prod", "loss"] = "rmse_prod"
    """Metric used to select the best model checkpoint."""

    loss: str | dict[str, LossSpecification] = "mse"
    """Loss function used for training."""
