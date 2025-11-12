"""
MACE
====

This architecture is a very thin wrapper around the official
MACE implementation, which is 
`hosted here <https://github.com/ACEsuit/mace>`_. Arbitrary heads
are added on top of MACE to predict an arbitrary number of targets
with arbitrary symmetry properties. These heads take as input the
output node features of MACE and pass them through a linear layer
(or MLP for invariant targets) to obtain the final predictions.

One important feature is that the architecture is ready to take
a pretrained MACE model file as input. The heads required to
predict the targets will be added on top of the MACE model, so
one can continue training for arbitrary targets. 
See :data:`ModelHypers.mace_model` for more details.
"""
from typing_extensions import TypedDict, NotRequired
from typing import Optional, Literal

from metatrain.utils.additive import FixedCompositionWeights
from metatrain.utils.loss import LossSpecification
from metatrain.utils.scaler import FixedScalerWeights

from metatrain.pet.modules.finetuning import FullFinetuneHypers

class ModelHypers(TypedDict):
    mace_model: Optional[str] = None
    mace_model_remove_scale_shift: bool = True
    mace_head_target: str = "energy"
    cutoff: float = 5.0
    num_radial_basis: int = 8
    """number of radial basis functions""" 
    num_cutoff_basis: int = 5
    """number of basis functions for smooth cutoff""" 
    max_ell: int = 3
    """highest \ell of spherical harmonics""" 
    interaction: Literal['RealAgnosticResidualInteractionBlock', 'RealAgnosticAttResidualInteractionBlock', 'RealAgnosticInteractionBlock', 'RealAgnosticDensityInteractionBlock', 'RealAgnosticDensityResidualInteractionBlock', 'RealAgnosticResidualNonLinearInteractionBlock'] = 'RealAgnosticResidualInteractionBlock'
    """name of interaction block""" 
    num_interactions: int = 2
    """number of interactions""" 
    hidden_irreps: Optional[str] = None
    """irreps for hidden node states""" 
    edge_irreps: Optional[str] = None
    """irreps for edge states""" 
    apply_cutoff: bool = True
    """apply cutoff to the radial basis functions before MLP""" 
    avg_num_neighbors: float = 1
    """normalization factor for the message""" 
    pair_repulsion: str = False
    """use pair repulsion term with ZBL potential""" 
    distance_transform: Optional[Literal['Agnesi', 'Soft']] = None
    """use distance transform for radial basis functions""" 
    correlation: int = 3
    """correlation order at each layer""" 
    gate: Optional[Literal['silu', 'tanh', 'abs']] = 'silu'
    """non linearity for last readout""" 
    interaction_first: Literal['RealAgnosticResidualInteractionBlock', 'RealAgnosticInteractionBlock', 'RealAgnosticDensityInteractionBlock', 'RealAgnosticDensityResidualInteractionBlock', 'RealAgnosticResidualNonLinearInteractionBlock'] = 'RealAgnosticResidualInteractionBlock'
    """name of interaction block""" 
    MLP_irreps: str = '16x0e'
    """hidden irreps of the MLP in last readout""" 
    radial_MLP: list[int] = [64, 64, 64]
    """width of the radial MLP""" 
    radial_type: Literal['bessel', 'gaussian', 'chebyshev'] = 'bessel'
    """type of radial basis functions""" 
    use_embedding_readout: bool = False
    """use embedding readout for the final output""" 
    use_last_readout_only: bool = False
    """use only the last readout for the final output""" 
    use_agnostic_product: bool = False
    """use element agnostic product""" 



class TrainerHypers(TypedDict):
    # Optimizer hypers (directly using MACE's scripts)
    optimizer: Literal['adam', 'adamw', 'schedulefree'] = 'adam'
    """Optimizer for parameter optimization"""
    learning_rate: float = 0.01 # Named "lr" in MACE
    """Learning rate of optimizer"""
    weight_decay: float = 5e-07
    """weight decay (L2 penalty)"""
    amsgrad: str = True
    """use amsgrad variant of optimizer"""
    beta: float = 0.9
    """Beta parameter for the optimizer"""

    # Scheduler hypers (directly using MACE's scripts)
    lr_scheduler: str = 'ReduceLROnPlateau' # Named "scheduler" in MACE
    """Type of scheduler"""
    lr_scheduler_gamma: float = 0.9993
    """Gamma of learning rate scheduler"""
    lr_factor: float = 0.8
    """Learning rate factor"""
    lr_scheduler_patience: int = 50 # Named "scheduler_patience" in MACE
    """Learning rate factor"""

    # General training parameters that are shared across architectures
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
    loss: str | dict[str, LossSpecification] = "mse"
    """This section describes the loss function to be used. See the
    :ref:`loss-functions` for more details."""

    finetune: NotRequired[FullFinetuneHypers]
    """Finetuning parameters for MetaMACE models that have been pretrained.

    See :ref:`fine-tuning` for more details.
    """