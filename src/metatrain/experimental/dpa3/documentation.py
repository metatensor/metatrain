"""
DPA3
====
"""

from typing import Literal, Optional

from typing_extensions import TypedDict

from metatrain.utils.additive import FixedCompositionWeights
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification


###########################
#  MODEL HYPERPARAMETERS  #
###########################


class RepflowHypers(TypedDict):
    n_dim: int = 128
    e_dim: int = 64
    a_dim: int = 32
    nlayers: int = 6
    e_rcut: float = 6.0
    e_rcut_smth: float = 5.3
    e_sel: int = 1200
    a_rcut: float = 4.0
    a_rcut_smth: float = 3.5
    a_sel: int = 300
    axis_neuron: int = 4
    skip_stat: bool = True
    a_compress_rate: int = 1
    a_compress_e_rate: int = 2
    a_compress_use_split: bool = True
    update_angle: bool = True
    # TODO: what are the options here
    update_style: str = "res_residual"
    update_residual: float = 0.1
    # TODO: what are the options here
    update_residual_init: str = "const"
    smooth_edge_update: bool = True
    use_dynamic_sel: bool = True
    sel_reduce_factor: float = 10.0


class DescriptorHypers(TypedDict):
    # TODO: what are the options here
    type: str = "dpa3"
    repflow: RepflowHypers = init_with_defaults(RepflowHypers)
    # TODO: what are the options here
    activation_function: str = "custom_silu:10.0"
    use_tebd_bias: bool = False
    # TODO: what are the options here
    precision: str = "float32"
    concat_output_tebd: bool = False


class FittingNetHypers(TypedDict):
    neuron: list[int] = [240, 240, 240]
    resnet_dt: bool = True
    seed: int = 1
    # TODO: what are the options here
    precision: str = "float32"
    # TODO: what are the options here
    activation_function: str = "custom_silu:10.0"
    # TODO: what are the options here
    type: str = "ener"
    numb_fparam: int = 0
    numb_aparam: int = 0
    dim_case_embd: int = 0
    trainable: bool = True
    rcond: Optional[float] = None
    atom_ener: list[float] = []
    use_aparam_as_mask: bool = False


class ModelHypers(TypedDict):
    """Hyperparameters for the DPA3 model."""

    type_map: list[str] = ["H", "C", "N", "O"]

    descriptor: DescriptorHypers = init_with_defaults(DescriptorHypers)
    fitting_net: FittingNetHypers = init_with_defaults(FittingNetHypers)


##############################
#  TRAINER HYPERPARAMETERS   #
##############################


class TrainerHypers(TypedDict):
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
    learning_rate: float = 0.001
    """Learning rate."""

    # TODO: update the scheduler or not
    scheduler_patience: int = 100
    scheduler_factor: float = 0.8

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
    # fixed_scaling_weights: FixedScalerWeights = {}
    # """Weights for target scaling.

    # This is passed to the ``fixed_weights`` argument of
    # :meth:`Scaler.train_model <metatrain.utils.scaler.scaler.Scaler.train_model>`,
    # see its documentation to understand exactly what to pass here.
    # """
    per_structure_targets: list[str] = []
    """Targets to calculate per-structure losses."""
    # num_workers: Optional[int] = None
    # """Number of workers for data loading. If not provided, it is set
    # automatically."""
    log_mae: bool = False
    """Log MAE alongside RMSE"""
    log_separate_blocks: bool = False
    """Log per-block error."""
    best_model_metric: Literal["rmse_prod", "mae_prod", "loss"] = "rmse_prod"
    """Metric used to select best checkpoint (e.g., ``rmse_prod``)"""

    loss: str | dict[str, LossSpecification] = "mse"
    """This section describes the loss function to be used. See the
    :ref:`loss-functions` for more details."""
