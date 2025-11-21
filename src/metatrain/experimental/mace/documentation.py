"""
MACE
====

This architecture is a very thin wrapper around the official MACE implementation, which
is `hosted here <https://github.com/ACEsuit/mace>`_. Arbitrary heads are added on top of
MACE to predict an arbitrary number of targets with arbitrary symmetry properties. These
heads take as input the output node features of MACE and pass them through a linear
layer (or MLP for invariant targets) to obtain the final predictions.

One important feature is that the architecture is ready to take a pretrained MACE model
file as input. The heads required to predict the targets will be added on top of the
MACE model, so one can continue training for arbitrary targets. See
:data:`ModelHypers.mace_model` for more details.
"""

from typing import Literal, Optional

from typing_extensions import NotRequired, TypedDict

from metatrain.pet.modules.finetuning import FullFinetuneHypers
from metatrain.utils.additive import FixedCompositionWeights
from metatrain.utils.loss import LossSpecification
from metatrain.utils.scaler import FixedScalerWeights


class ModelHypers(TypedDict):
    mace_model: Optional[str] = None
    """Path to a pretrained MACE model file.

    For example, this can be `a foundation MACE model
    <https://github.com/ACEsuit/mace-foundations>`_. If not provided, a new MACE model
    will be initialized from scratch using the rest of hyperparameters of the
    architecture.
    """
    mace_model_remove_scale_shift: bool = True
    """Whether to remove the scale and shift block from the
    pretrained MACE model (if provided).

    If the loaded model is a ``ScaleShiftMACE``, it contains a block that scales and
    shifts the outputs of MACE. In metatrain, these things are handled by the ``Scaler``
    and ``CompositionModels`` class, so it is probably more natural to continue training
    without this block.

    However, one might be using ``mtt train`` with 0 epochs simply to be able to export
    a MACE model, in which case it probably makes more sense to keep the scale and shift
    block.
    """
    mace_head_target: str = "energy"
    """Target to which the MACE head is related.

    ``metatrain`` adds arbitrary heads on top of MACE to predict arbitrary targets.
    However, MACE models have themselves a head. This hyperparameter specifies which
    metatrain target corresponds to the MACE head. For this target, no new head will be
    added, and the output of MACE's head will be used directly.
    """
    cutoff: float = 5.0
    """Cutoff radius for neighbor search.

    This should be set to a value after which most of the interactions between atoms is
    expected to be negligible. A lower cutoff will lead to faster models.

    This is passed to MACE's ``r_max`` argument.
    """
    num_radial_basis: int = 8
    r"""number of radial basis functions"""
    num_cutoff_basis: int = 5
    r"""number of basis functions for smooth cutoff"""
    max_ell: int = 3
    r"""highest \ell of spherical harmonics"""
    interaction: Literal[
        "RealAgnosticResidualInteractionBlock",
        "RealAgnosticAttResidualInteractionBlock",
        "RealAgnosticInteractionBlock",
        "RealAgnosticDensityInteractionBlock",
        "RealAgnosticDensityResidualInteractionBlock",
        "RealAgnosticResidualNonLinearInteractionBlock",
    ] = "RealAgnosticResidualInteractionBlock"
    r"""name of interaction block"""
    num_interactions: int = 2
    r"""number of interactions"""
    hidden_irreps: Optional[str] = "128x0e + 128x1o + 128x2e"
    r"""irreps for hidden node states"""
    edge_irreps: Optional[str] = None
    r"""irreps for edge states"""
    apply_cutoff: bool = True
    r"""apply cutoff to the radial basis functions before MLP"""
    avg_num_neighbors: float = 1
    r"""normalization factor for the message"""
    pair_repulsion: bool = False
    r"""use pair repulsion term with ZBL potential"""
    distance_transform: Optional[Literal["Agnesi", "Soft"]] = None
    r"""use distance transform for radial basis functions"""
    correlation: int = 3
    r"""correlation order at each layer"""
    gate: Optional[Literal["silu", "tanh", "abs"]] = "silu"
    r"""non linearity for last readout"""
    interaction_first: Literal[
        "RealAgnosticResidualInteractionBlock",
        "RealAgnosticInteractionBlock",
        "RealAgnosticDensityInteractionBlock",
        "RealAgnosticDensityResidualInteractionBlock",
        "RealAgnosticResidualNonLinearInteractionBlock",
    ] = "RealAgnosticResidualInteractionBlock"
    r"""name of interaction block"""
    MLP_irreps: str = "16x0e"
    r"""hidden irreps of the MLP in last readout"""
    radial_MLP: list[int] = [64, 64, 64]
    r"""width of the radial MLP"""
    radial_type: Literal["bessel", "gaussian", "chebyshev"] = "bessel"
    r"""type of radial basis functions"""
    use_embedding_readout: bool = False
    r"""use embedding readout for the final output"""
    use_last_readout_only: bool = False
    r"""use only the last readout for the final output"""
    use_agnostic_product: bool = False
    r"""use element agnostic product"""


class TrainerHypers(TypedDict):
    # Optimizer hypers (directly using MACE's scripts)
    optimizer: Literal["adam", "adamw", "schedulefree"] = "adam"
    """Optimizer for parameter optimization"""
    learning_rate: float = 0.01  # Named "lr" in MACE
    """Learning rate of optimizer"""
    weight_decay: float = 5e-07
    """weight decay (L2 penalty)"""
    amsgrad: bool = True
    """use amsgrad variant of optimizer"""
    beta: float = 0.9
    """Beta parameter for the optimizer"""

    # Scheduler hypers (directly using MACE's scripts)
    lr_scheduler: str = "ReduceLROnPlateau"  # Named "scheduler" in MACE
    """Type of scheduler"""
    lr_scheduler_gamma: float = 0.9993
    """Gamma of learning rate scheduler"""
    lr_factor: float = 0.8
    """Learning rate factor"""
    lr_scheduler_patience: int = 50  # Named "scheduler_patience" in MACE
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
    <metatrain.utils.additive.composition.CompositionModel.train_model>`, see its
    documentation to understand exactly what to pass here.
    """
    remove_composition_contribution: bool = True
    """Whether to remove the atomic composition contribution from the
    targets by fitting a linear model to the training data before training the neural
    network."""
    fixed_scaling_weights: FixedScalerWeights = {}
    """Weights for target scaling.

    This is passed to the ``fixed_weights`` argument of :meth:`Scaler.train_model
    <metatrain.utils.scaler.scaler.Scaler.train_model>`, see its documentation to
    understand exactly what to pass here.
    """
    per_structure_targets: list[str] = []
    """Targets to calculate per-structure losses."""
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

    finetune: NotRequired[FullFinetuneHypers]
    """Finetuning parameters for MetaMACE models that have been pretrained.

    See :ref:`fine-tuning` for more details.
    """
