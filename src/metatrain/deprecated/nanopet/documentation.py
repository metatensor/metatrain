"""
NanoPET (deprecated)
======================

.. image:: https://codecov.io/gh/metatensor/metatrain/branch/main/graph/badge.svg?flag=coverage_nanopet
   :target: https://codecov.io/gh/metatensor/metatrain/tree/main/src/metatrain/deprecated/nanopet

.. warning::

  This is a **deprecated model**. You should not use it for anything important, and
  support for it will be removed in future versions of metatrain. Please use the
  :ref:`PET model <architecture-pet>` instead.
"""

from typing import Literal, Optional

from typing_extensions import TypedDict

from metatrain.utils.additive import FixedCompositionWeights
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.long_range import LongRangeHypers
from metatrain.utils.loss import LossSpecification
from metatrain.utils.scaler import FixedScalerWeights


class ModelHypers(TypedDict):
    """Hyperparameters for the NanoPET model."""

    cutoff: float = 5.0
    """Cutoff radius for neighbor search.

    This should be set to a value after which most of the interactions
    between atoms is expected to be negligible. A lower cutoff will lead
    to faster models.
    """
    cutoff_width: float = 0.5
    """Width of the smoothing function at the cutoff"""
    d_pet: int = 128
    """Dimension of the edge features.

    This hyperparameters controls width of the neural network. In general,
    increasing it might lead to better accuracy, especially on larger datasets, at the
    cost of increased training and evaluation time.
    """
    num_heads: int = 4
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
    heads: dict[str, Literal["linear", "mlp"]] = {}
    """The type of head ("linear" or "mlp") to use for each target (e.g.
    ``heads: {"energy": "linear", "mtt::dipole": "mlp"}``). All omitted targets will
    use a MLP (multi-layer perceptron) head. MLP heads consist of two hidden layers
    with dimensionality ``d_pet``."""
    zbl: bool = False
    """Use ZBL potential for short-range repulsion"""
    long_range: LongRangeHypers = init_with_defaults(LongRangeHypers)
    """Long-range Coulomb interactions parameters."""


class TrainerHypers(TypedDict):
    """Hyperparameters for training NanoPET models."""

    distributed: bool = False
    """Whether to use distributed training"""
    distributed_port: int = 39591
    """Port for DDP communication"""
    batch_size: int = 16
    """The number of samples to use in each batch of training. This
    hyperparameter controls the tradeoff between training speed and memory usage. In
    general, larger batch sizes will lead to faster training, but might require more
    memory."""
    num_epochs: int = 10000
    """Number of epochs."""
    learning_rate: float = 3e-4
    """Learning rate."""
    scheduler_patience: int = 100
    """Patience for the learning rate scheduler."""
    scheduler_factor: float = 0.8
    """Factor to reduce the learning rate by"""
    log_interval: int = 10
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
    log_mae: bool = False
    """Log MAE alongside RMSE"""
    log_separate_blocks: bool = False
    """Log per-block error."""
    best_model_metric: Literal["rmse_prod", "mae_prod", "loss"] = "rmse_prod"
    """Metric used to select best checkpoint (e.g., ``rmse_prod``)"""
    loss: str | dict[str, LossSpecification | str] = "mse"
    """This section describes the loss function to be used. See the
    :ref:`loss-functions` for more details."""
