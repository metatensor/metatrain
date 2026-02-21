"""
Graph2Mat
=========

Interface of ``Graph2Mat`` to all architectures in ``metatrain``.
"""

from typing import Literal, Optional

from typing_extensions import TypedDict

from metatrain.utils.loss import LossSpecification


class ModelHypers(TypedDict):
    featurizer_architecture: dict
    """Architecture that provides the features for graph2mat.
    
    This hyperparameter can contain the full specification for the
    architecture, i.e. everything that goes inside the ``architecture``
    field of a normal training run for that architecture.
    """
    basis_yaml: str = "."
    """Yaml file with the full basis specification for graph2mat.
    
    This file contains a list, with each item being a dictionary
    to initialize a ``graph2mat.PointBasis`` object.
    """
    basis_grouping: Literal["point_type", "basis_shape", "max"] = "point_type"
    """The way in which graph2mat should group basis (to reduce the number of heads)"""
    node_hidden_irreps: str = "20x0e+20x1o+20x2e"
    """Irreps to ask for to the featurizer (per atom).
    
    Graph2Mat will take these features as input.
    """
    edge_hidden_irreps: str = "10x0e+10x1o+10x2e"
    """Hidden irreps for the edges inside graph2mat"""


class TrainerHypers(TypedDict):
    # Optimizer hypers
    optimizer: str = "Adam"
    """Optimizer for parameter optimization.
    
    We just take the class from torch.optim by name, so make
    sure it is a valid torch optimizer (including possible
    uppercase/lowercase differences).
    """
    optimizer_kwargs: dict = {"lr": 0.01}
    """Keyword arguments to pass to the optimizer.
    
    These will depend on the optimizer chosen.
    """

    # LR scheduler hypers
    lr_scheduler: Optional[str] = "ReduceLROnPlateau"  # Named "scheduler" in MACE
    """Learning rate scheduler to use.
    
    We just take the class from torch.optim.lr_scheduler by name, so make
    sure it is a valid torch scheduler (including possible
    uppercase/lowercase differences).

    None means no scheduler will be used.
    """
    lr_scheduler_kwargs: dict = {}
    """Keyword arguments to pass to the learning rate scheduler.
    
    These will depend on the scheduler chosen.
    """

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
    """Maximum gradient norm value"""
    loss: str | dict[str, LossSpecification] = "mse"
    """This section describes the loss function to be used. See the
    :ref:`loss-functions` for more details."""
