"""
Graph2Mat
=========

Interface of ``Graph2Mat`` to all architectures in ``metatrain``.
"""

from typing import Literal, Optional

from typing_extensions import TypedDict

from metatrain.utils.loss import LossSpecification
from metatrain.scaler.documentation import FixedScalerWeights
from metatrain.composition.documentation import FixedCompositionWeights


class MatrixSpecification(TypedDict):
    nodes: str
    """Name of the target that contains the node features
    of the matrix."""
    edges: Optional[str] = None
    """Name of the target that contains the edge features of the matrix."""
    basis_grouping: Literal["point_type", "basis_shape", "max"] = "point_type"
    """The way in which graph2mat should group basis (to reduce the number of heads)"""
    symmetric: bool = False
    """Whether the matrix is symmetric"""
    self_blocks_symmetry: Optional[str] = None
    node_operation: Literal["linear", "tsq"] = "tsq"
    edge_operation: Literal["linear", "simple"] = "simple"
    preprocessing_edges: Literal["two_center_message", "mace_node_message"] = (
        "mace_edge_message"
    )
    preprocessing_nodes: Optional[str] = None
    edge_cutoff: float
    learn_log: bool = False
    """Whether to learn the log of the matrix instead of the matrix itself."""


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
    node_hidden_irreps: str = "20x0e+20x1o+20x2e"
    """Irreps to ask for to the featurizer (per atom).
    
    Graph2Mat will take these features as input.
    """
    edge_hidden_irreps: str = "10x0e+10x1o+10x2e"
    """Hidden irreps for the edges inside graph2mat"""
    edge_max_ell: int = 2
    """Maximum ell for the embedding of edge directions into spherical harmonics.
    
    This should be at least the maximum ell in the edge target that you want to
    predict.
    """
    matrices: dict[str, dict] = {}
    """Dictionary with the specification of the matrices to be predicted.
    
    The keys of the dictionary are the names of the matrices, and the
    values specify which nodes and edges to use for each matrix.
    """
    edge_composition: Optional[str] = None
    """Path to a checkpoint for a pretrained edge composition model."""


class TrainerHypers(TypedDict):
    profile: Optional[str] = None
    """If not None, the last epoch of training will be profiled and
    the trace will be saved to this path as json."""

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
    scale_targets: bool = True
    """Normalize targets to unit std during training."""
    fixed_scaling_weights: FixedScalerWeights = {}
    """Weights for target scaling.

    This is passed to the ``fixed_weights`` argument of :meth:`Scaler.train_model
    <metatrain.utils.scaler.scaler.Scaler.train_model>`, see its documentation to
    understand exactly what to pass here.

    .. note::
        If a MACE model is loaded through the ``mace_model`` hyperparameter, the
        scales in the MACE model are used by default for the target
        indicated in ``mace_head_target``. If you want to override them, you need
        to set explicitly the baselines for that target in this hyperparameter.
    """
    atomic_baseline: FixedCompositionWeights | str = {}
    """The baselines for each target.

    By default, ``metatrain`` will fit a linear model (:class:`CompositionModel
    <metatrain.composition.CompositionModel>`) to compute the least squares
    baseline for each atomic species for each target.

    However, this hyperparameter allows you to provide your own baselines,
    either as a dictionary or as a path to a pre-trained composition model
    checkpoint. The value of the hyperparameter should either be:

    - a dictionary where the keys are the target names, and the values are
      either (1) a single baseline to be used for all atomic types, or
      (2) a dictionary mapping atomic types to their baselines.
    - a string path to a ``.ckpt`` file from a pre-trained composition model.

    For example:

    - ``atomic_baseline: {"energy": {1: -0.5, 6: -10.0}}`` will fix the energy
      baseline for hydrogen (Z=1) to -0.5 and for carbon (Z=6) to -10.0, while
      fitting the baselines for the energy of all other atomic types, as well
      as fitting the baselines for all other targets.
    - ``atomic_baseline: {"energy": -5.0}`` will fix the energy baseline for
      all atomic types to -5.0.
    - ``atomic_baseline: {"mtt:dos": 0.0}`` sets the baseline for the "mtt:dos"
      target to 0.0, effectively disabling the atomic baseline for that target.
    - ``atomic_baseline: "/path/to/model.ckpt"`` loads a pre-trained
      composition model checkpoint, overriding the default least-squares fit.

    This atomic baseline is substracted from the targets during training, which
    avoids the main model needing to learn atomic contributions, and likely makes
    training easier. When the model is used in evaluation mode, the atomic baseline
    is added on top of the model predictions automatically.

    .. note::
        This atomic baseline is a per-atom contribution. Therefore, if the property
        you are predicting is a sum over all atoms (e.g., total energy), the
        contribution of the atomic baseline to the total property will be the
        atomic baseline multiplied by the number of atoms of that type in the
        structure.
    """
    best_model_metric: Literal["rmse_prod", "mae_prod", "loss"] = "mae_prod"
    """Metric used to select best checkpoint (e.g., ``rmse_prod``)"""
    grad_clip_norm: float = 1.0
    """Maximum gradient norm value"""
    loss: str | dict[str, LossSpecification] = "mse"
    """This section describes the loss function to be used. See the
    :ref:`loss-functions` for more details."""
