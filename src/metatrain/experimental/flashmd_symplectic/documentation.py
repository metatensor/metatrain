"""
Symplectic FlashMD
==================

The symplectic variant of :ref:`FlashMD <arch-flashmd>`.

{{SECTION_INSTALLATION}}

{{SECTION_DEFAULT_HYPERS}}
"""

from typing import Literal, Optional

from typing_extensions import TypedDict

from metatrain.pet.modules.finetuning import FinetuneHypers, NoFinetuneHypers
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.long_range import LongRangeHypers
from metatrain.utils.loss import LossSpecification
from metatrain.utils.scaler import FixedScalerWeights


###########################
#  MODEL HYPERPARAMETERS  #
###########################


class ModelHypers(TypedDict):
    """Hyperparameters for the FlashMD model."""

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
    d_node: int = 512
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
    num_gnn_layers: int = 3
    """The number of graph neural network layers.

    In general, decreasing this hyperparameter to 1 will lead to much faster models,
    at the expense of accuracy. Increasing it may or may not lead to better accuracy,
    depending on the dataset, at the cost of increased training and evaluation time.
    """
    normalization: Literal["RMSNorm", "LayerNorm"] = "RMSNorm"
    """Layer normalization type."""
    activation: Literal["SiLU", "SwiGLU"] = "SwiGLU"
    """Activation function."""
    attention_temperature: float = 1.0
    """The temperature scaling factor for attention scores."""
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
    long_range: LongRangeHypers = init_with_defaults(LongRangeHypers)
    """Long-range Coulomb interactions parameters."""


##############################
#  TRAINER HYPERPARAMETERS   #
##############################


class TrainerHypers(TypedDict):
    """Hyperparameters for training symplectic FlashMD models."""

    timestep: Optional[float] = None
    """The time interval (in fs) between the current and the future positions
    and momenta that the model must predict. This option is not used in the
    training, but it is registered in the model and it will be used to validate
    that the timestep used during inference in MD engines is the same as the
    one used during training. This hyperparameter must be provided by the user."""
    masses: dict[int, float] = {}
    """
    A dictionary mapping atomic species to their masses (in atomic mass
    units).

    Indeed, it should be noted that FlashMD models, as implemented in metatrain,
    are not transferable across different isotopes. The masses are not used during
    training, but they are registered in the model and they will be used during
    inference to validate that the masses used in MD engines are the same as the
    ones used during training. If not provided, masses from the ``ase.data``
    module will be used. These correspond to masses averaged over the natural
    isotopic abundance of each element.
    """
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
    learning_rate: float = 3e-4
    """Learning rate."""
    weight_decay: Optional[float] = None
    """Weight decay (L2 penalty)."""
    log_interval: int = 1
    """Interval to log metrics."""
    checkpoint_interval: int = 100
    """Interval to save checkpoints."""
    scale_targets: bool = False
    """Normalize targets to unit std during training."""
    fixed_scaling_weights: FixedScalerWeights = {}
    """Weights for target scaling.

    This is passed to the ``fixed_weights`` argument of
    :meth:`Scaler.train_model <metatrain.utils.scaler.scaler.Scaler.train_model>`,
    see its documentation to understand exactly what to pass here.

    .. warning::

        For symplectic FlashMD, the scaling weights for the variables
        (``mtt::delta_q`` and ``mtt::delta_p``) must be identical. This is because
        they are both derivatives of the same generating function :math:`S^3`.
        Scaling them differently would lead to inconsistencies.
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
    best_model_metric: Literal["rmse_prod", "mae_prod", "loss"] = "rmse_prod"
    """Metric used to select best checkpoint (e.g., ``rmse_prod``)"""
    grad_clip_norm: float = 1.0
    """Maximum gradient norm value, by default inf (no clipping)"""
    loss: str | dict[str, LossSpecification | str] = "mse"
    """This section describes the loss function to be used. See the
    :ref:`loss-functions` for more details."""
    batch_atom_bounds: list[Optional[int]] = [None, None]
    """Bounds for the number of atoms per batch as [min, max]. Batches with atom
    counts outside these bounds will be skipped during training. Use ``None`` for
    either value to disable that bound. This is useful for preventing out-of-memory
    errors and ensuring consistent computational load. Default: ``[None, None]``."""

    finetune: NoFinetuneHypers | FinetuneHypers = {
        "read_from": None,
        "method": "full",
        "config": {},
        "inherit_heads": {},
    }
    """Parameters for fine-tuning trained FlashMD models."""
