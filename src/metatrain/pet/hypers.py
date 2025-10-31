# mypy: disable-error-code=misc
# We ignore misc errors in this file because TypedDict
# with default values is not allowed by mypy.
from typing import Literal, NotRequired, Optional

from typing_extensions import TypedDict

from metatrain.utils.hypers import (
    CompositionWeightsDict,
    ScalingWeightsDict,
    init_with_defaults,
)
from metatrain.utils.loss import LossSpecification


###########################
#  MODEL HYPERPARAMETERS  #
###########################

class LongRangeHypers(TypedDict):
    """In some systems and datasets, enabling long-range Coulomb interactions
    might be beneficial for the accuracy of the model and/or
    its physical correctness."""

    enable: bool = False
    """Toggle for enabling long-range interactions"""
    use_ewald: bool = False
    """Use Ewald summation. If False, P3M is used"""
    smearing: float = 1.4
    """Smearing width in Fourier space"""
    kspace_resolution: float = 1.33
    """Resolution of the reciprocal space grid"""
    interpolation_nodes: int = 5
    """Number of grid points for interpolation (for PME only)"""


class PETHypers(TypedDict):
    """Hyperparameters for the PET model."""

    cutoff: float = 4.5
    """Cutoff radius for neighbor search.

    This should be set to a value after which most of the interactions 
    between atoms is expected to be negligible. A lower cutoff will lead
    to faster models.
    """
    cutoff_width: float = 0.2
    """Width of the smoothing function at the cutoff"""
    d_pet: int = 128
    """Dimension of the edge features.
    
    This hyperparameters controls width of the neural network. In general,
    increasing it might lead to better accuracy, especially on larger datasets, at the
    cost of increased training and evaluation time.
    """
    d_head: int = 128
    """Dimension of the attention heads."""
    d_node: int = 256
    """Dimension of the node features.

    Increasing this hyperparameter might lead to better accuracy, 
    with a relatively small increase in inference time.
    """
    d_feedforward: int = 256
    """Dimension of the feedforward network in the attention layer."""
    num_heads: int = 8
    """Attention heads per attention layer."""
    num_attention_layers: int = 2
    """The number of attention layers in each layer of the graph
    neural network. Depending on the dataset, increasing this hyperparameter might 
    lead to better accuracy, at the cost of increased training and evaluation time.
    """
    num_gnn_layers: int = 2
    """The number of graph neural network layers. 
    
    In general, decreasing this hyperparameter to 1 will lead to much faster models,
    at the expense of accuracy. Increasing it may or may not lead to better accuracy,
    depending on the dataset, at the cost of increased training and evaluation time.
    """
    normalization: Literal["RMSNorm", "LayerNorm"] = "RMSNorm"
    """Layer normalization type."""
    activation: Literal["SiLU", "SwiGLU"] = "SwiGLU"
    """Activation function."""
    transformer_type: Literal["PreLN", "PostLN"] = "PreLN"
    """The order in which the layer normalization and attention
    are applied in a transformer block. Available options are ``PreLN``
    (normalization before attention) and ``PostLN`` (normalization after attention)."""
    featurizer_type: Literal["residual", "feedforward"] = "feedforward"
    """Implementation of the featurizer of the model to use. Available
    options are ``residual`` (the original featurizer from the PET paper, that uses
    residual connections at each GNN layer for readout) and ``feedforward`` (a modern
    version that uses the last representation after all GNN iterations for readout).
    Additionally, the feedforward version uses bidirectional features flow during the
    message passing iterations, that favors features flowing from atom ``i`` to atom
    ``j`` to be not equal to the features flowing from atom ``j`` to atom ``i``."""
    zbl: bool = False
    """Use ZBL potential for short-range repulsion"""
    long_range: LongRangeHypers = init_with_defaults(LongRangeHypers)
    """Long-range Coulomb interactions parameters."""

##############################
#  TRAINER HYPERPARAMETERS   #
##############################

class LoRaFinetuneConfig(TypedDict):
    """Configuration for LoRA finetuning strategy."""

    rank: int
    """Rank of the LoRA matrices."""
    alpha: float
    """Scaling factor for the LoRA matrices."""


class HeadsFinetuneConfig(TypedDict):
    """Configuration for heads finetuning strategy."""

    head_modules: list[str]
    """List of module name prefixes for the prediction heads to finetune."""
    last_layer_modules: list[str]
    """List of module name prefixes for the last layers to finetune."""


class FinetuneHypers(TypedDict):
    """Hyperparameters for finetuning PET models."""

    read_from: str
    """Path to the pretrained model checkpoint."""
    method: Literal["full", "lora", "heads"]
    """Finetuning method to use. Available methods are:
    - ``full``: finetune all model parameters.
    - ``lora``: inject LoRA layers and finetune only them.
    - ``heads``: finetune only the prediction heads of the model.
    """
    config: LoRaFinetuneConfig | HeadsFinetuneConfig | None
    """Configuration for the selected finetuning method."""


class PETTrainerHypers(TypedDict):
    """Hyperparameters for training PET models."""

    distributed: bool = False
    """Whether to use distributed training"""
    distributed_port: int = 39591
    """Port for DDP communication"""
    batch_size: int = 16
    """The number of samples to use in each batch of training. This
    hyperparameter controls the tradeoff between training speed and memory usage. In
    general, larger batch sizes will lead to faster training, but might require more
    memory."""
    num_epochs: int = 1000
    """Number of epochs."""
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
    fixed_composition_weights: CompositionWeightsDict = {}
    """Weights for atomic contributions."""
    fixed_scaling_weights: ScalingWeightsDict = {}

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
    loss: str | dict[str, LossSpecification] = "mse"
    """This section describes the loss function to be used. See the
    :ref:`loss-functions` for more details."""

    finetune: NotRequired[FinetuneHypers]
    """Finetuning parameters for PET models pretrained on large datasets."""
