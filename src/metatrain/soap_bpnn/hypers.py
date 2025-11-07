# mypy: disable-error-code=misc
# We ignore misc errors in this file because TypedDict
# with default values is not allowed by mypy.
from typing import Literal, Optional

from typing_extensions import TypedDict

from metatrain.utils.additive.composition import FixedCompositionWeights
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.long_range import LongRangeHypers
from metatrain.utils.loss import LossSpecification
from metatrain.utils.scaler.scaler import FixedScalerWeights


class SOAPCutoffConfig(TypedDict):
    """Cutoff configuration for the SOAP descriptor."""

    radius: float = 5.0
    """Should be set to a value after which most of interatomic is expected
    to be negligible. Note that the values should be defined in the position
    units of your dataset."""
    width: float = 0.5
    """The radial cutoff of atomic environments is performed smoothly, over
    another distance defined by this parameter."""


class SOAPConfig(TypedDict):
    """Configuration for the SOAP descriptors."""

    max_angular: int = 6
    """Maximum angular channels of the spherical harmonics when
    computing the SOAP descriptors."""
    max_radial: int = 7
    """Maximum radial channels of the spherical harmonics when
    computing the SOAP descriptors."""

    cutoff: SOAPCutoffConfig = init_with_defaults(SOAPCutoffConfig)
    """Determines the cutoff routine of the atomic environment."""


class BPNNConfig(TypedDict):
    """Configuration for the BPNN architecture."""

    num_hidden_layers: int = 2
    """Controls the depth of the neural network. Increasing this generally leads
    to better accuracy from the increased descriptivity, but comes at the cost
    of increased training and evaluation time."""

    num_neurons_per_layer: int = 32
    """Controls the width of the neural network. Increasing this generally leads
    to better accuracy from the increased descriptivity, but comes at the cost
    of increased training and evaluation time."""

    layernorm: bool = True
    """Whether to use layer normalization before the neural network. Setting this
    hyperparameter to false will lead to slower convergence of training, but
    might lead to better generalization outside of the training set distribution.
    """


class SOAPBPNNHypers(TypedDict):
    """Hyperparameters for the SOAP + BPNN architecture."""

    soap: SOAPConfig = init_with_defaults(SOAPConfig)
    """Configuration of the SOAP descriptors."""

    bpnn: BPNNConfig = init_with_defaults(BPNNConfig)
    """Configuration of the neural network architecture."""

    add_lambda_basis: bool = True
    """This boolean parameter controls whether or not to add a spherical
    expansion term of the same angular order as the targets, when they are
    tensorial."""

    heads: dict[str, Literal["mlp", "linear"]] = {}
    """The type of head (“linear” or “mlp”) to use for each target
    (e.g. heads: {"energy": "linear", "mtt::dipole": "mlp"}). All omitted
    targets will use a MLP (multi-layer perceptron) head. MLP heads consists
    of one hidden layer with as many neurons as the
    SOAP-BPNN (see ``BPNNConfig.num_neurons_per_layer``)."""

    zbl: bool = False
    """Whether to use the ZBL short-range repulsion as the baseline for the model.
    May be needed to achieve better description at the close-contact, repulsive regime.
    """

    long_range: LongRangeHypers = init_with_defaults(LongRangeHypers)
    """Parameters related to long-range interactions.

    May be needed to describe important long-range effects not captured by
    the short-range SOAP-BPNN model"""


class SOAPBPNNTrainerHypers(TypedDict):
    """Hyperparameters for training SOAP BPNN models."""

    distributed: bool = False
    """Whether to use distributed training"""
    distributed_port: int = 39591
    """Port for DDP communication"""
    batch_size: int = 8
    """The number of samples to use in each batch of training. This
    hyperparameter controls the tradeoff between training speed and memory usage. In
    general, larger batch sizes will lead to faster training, but might require more
    memory."""
    num_epochs: int = 100
    """Number of epochs."""
    warmup_fraction: float = 0.01
    """Fraction of training steps used for learning rate warmup."""
    learning_rate: float = 1e-3
    """Learning rate."""
    log_interval: int = 5
    """Interval to log metrics."""
    checkpoint_interval: int = 25
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
    log_mae: bool = False
    """Log MAE alongside RMSE"""
    log_separate_blocks: bool = False
    """Log per-block error."""
    best_model_metric: Literal["rmse_prod", "mae_prod", "loss"] = "rmse_prod"
    """Metric used to select best checkpoint (e.g., ``rmse_prod``)"""
    loss: str | dict[str, LossSpecification] = "mse"
    """This section describes the loss function to be used. See the
    :ref:`loss-functions` for more details."""
