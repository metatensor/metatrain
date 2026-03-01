"""
DPA3 (experimental)
======================

This is an interface to the DPA3 (Deep Potential Attention 3) architecture
:footcite:p:`dpa3_2025` implemented in `deepmd-kit
<https://github.com/deepmodeling/deepmd-kit>`_.

DPA3 extends the DPA series with a Line Graph representation and the RepFlow
framework, enabling richer many-body interactions through joint edge-angle
message passing.  See the `paper <https://arxiv.org/abs/2506.01686>`_ and the
`deepmd-kit documentation <https://docs.deepmodeling.com/projects/deepmd/>`_
for further details.

.. note::

   The ``type_map`` required by deepmd-kit is derived automatically from the
   atomic numbers present in the dataset; it is *not* a user-facing
   hyperparameter.

{{SECTION_INSTALLATION}}

{{SECTION_DEFAULT_HYPERS}}

Tuning hyperparameters
----------------------

The most impactful hyperparameters (roughly in decreasing order of importance):

.. container:: mtt-hypers-remove-classname

  .. autoattribute:: {{model_hypers_path}}.descriptor
      :no-index:

  .. autoattribute:: {{trainer_hypers_path}}.learning_rate
      :no-index:

  .. autoattribute:: {{trainer_hypers_path}}.batch_size
      :no-index:

Increasing ``descriptor.repflow.nlayers`` typically improves accuracy at the
cost of training time.  ``descriptor.repflow.e_rcut`` controls the interaction
range and should be chosen based on the physical system.  Reduce ``e_sel`` and
``a_sel`` for faster iteration on small systems.

References
----------

.. footbibliography::

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
    """RepFlow descriptor block parameters."""

    n_dim: int = 128
    """Node feature dimension."""
    e_dim: int = 64
    """Edge feature dimension."""
    a_dim: int = 32
    """Angle feature dimension."""
    nlayers: int = 6
    """Number of RepFlow interaction layers."""
    e_rcut: float = 6.0
    """Edge (pair) cutoff radius in length units."""
    e_rcut_smth: float = 5.3
    """Start of cosine smoothing for the edge cutoff."""
    e_sel: int = 1200
    """Maximum number of edge neighbors per atom."""
    a_rcut: float = 4.0
    """Angle (triplet) cutoff radius in length units."""
    a_rcut_smth: float = 3.5
    """Start of cosine smoothing for the angle cutoff."""
    a_sel: int = 300
    """Maximum number of angle neighbors per atom."""
    axis_neuron: int = 4
    """Number of axis neurons in the embedding network."""
    skip_stat: bool = True
    """Skip statistics computation (use pretrained stats)."""
    a_compress_rate: int = 1
    """Compression rate for angle features."""
    a_compress_e_rate: int = 2
    """Compression rate for angle-edge features."""
    a_compress_use_split: bool = True
    """Use split compression for angle features."""
    update_angle: bool = True
    """Update angle features at each layer."""
    update_style: str = "res_residual"
    """Residual update style. Options: ``"res_residual"``, ``"res_avg"``."""
    update_residual: float = 0.1
    """Residual scaling factor for updates."""
    update_residual_init: str = "const"
    """Initialisation for the residual scaling. Options: ``"const"``,
    ``"norm"``."""
    smooth_edge_update: bool = True
    """Apply smooth cutoff function to edge updates."""
    use_dynamic_sel: bool = True
    """Dynamically adjust neighbor selection at runtime."""
    sel_reduce_factor: float = 10.0
    """Reduction factor for dynamic neighbor selection."""


class DescriptorHypers(TypedDict):
    """Descriptor hyperparameters wrapping the RepFlow block."""

    type: str = "dpa3"
    """Descriptor type identifier used by deepmd-kit."""
    repflow: RepflowHypers = init_with_defaults(RepflowHypers)
    """RepFlow block parameters."""
    activation_function: str = "custom_silu:10.0"
    """Activation function. Format: ``"name"`` or ``"name:param"``.
    Supported names include ``"tanh"``, ``"gelu"``, ``"custom_silu"``."""
    use_tebd_bias: bool = False
    """Add bias to the type embedding."""
    precision: str = "float32"
    """Floating-point precision for the descriptor. ``"float32"`` or
    ``"float64"``.  This controls the internal precision of deepmd-kit's
    descriptor computation.  For mixed-precision training, set this
    independently of ``fitting_net.precision``; for uniform precision, set
    both to the same value and match ``base_precision`` accordingly."""
    concat_output_tebd: bool = False
    """Concatenate type embedding to descriptor output."""


class FittingNetHypers(TypedDict):
    """Fitting network hyperparameters."""

    neuron: list[int] = [240, 240, 240]
    """Hidden layer sizes for the fitting network."""
    resnet_dt: bool = True
    """Use a ResNet-style time step in each hidden layer."""
    seed: int = 1
    """Random seed for weight initialisation."""
    precision: str = "float32"
    """Floating-point precision for the fitting network. ``"float32"`` or
    ``"float64"``.  Can differ from ``descriptor.precision`` for
    mixed-precision training."""
    activation_function: str = "custom_silu:10.0"
    """Activation function (same format as the descriptor)."""
    type: str = "ener"
    """Fitting type. ``"ener"`` for energy fitting."""
    numb_fparam: int = 0
    """Number of frame-level parameters."""
    numb_aparam: int = 0
    """Number of atom-level parameters."""
    dim_case_embd: int = 0
    """Dimension of the case embedding (multi-task)."""
    trainable: bool = True
    """Whether fitting network weights are trainable."""
    rcond: Optional[float] = None
    """Cutoff for pseudo-inverse in linear fitting."""
    atom_ener: list[float] = []
    """Per-type atomic energy offsets."""
    use_aparam_as_mask: bool = False
    """Treat atom-level parameters as a mask."""


class ModelHypers(TypedDict):
    """Hyperparameters for the DPA3 model.

    The ``type_map`` needed by deepmd-kit is derived automatically from the
    dataset's atomic numbers and should not be set manually.
    """

    descriptor: DescriptorHypers = init_with_defaults(DescriptorHypers)
    """Descriptor configuration (RepFlow block and related settings)."""
    fitting_net: FittingNetHypers = init_with_defaults(FittingNetHypers)
    """Fitting network configuration."""


##############################
#  TRAINER HYPERPARAMETERS   #
##############################


class TrainerHypers(TypedDict):
    """Hyperparameters for training DPA3 models."""

    distributed: bool = False
    """Whether to use distributed training."""
    distributed_port: int = 39591
    """Port for DDP communication."""
    batch_size: int = 8
    """The number of samples to use in each batch of training. This
    hyperparameter controls the tradeoff between training speed and memory usage. In
    general, larger batch sizes will lead to faster training, but might require more
    memory."""
    num_epochs: int = 100
    """Number of epochs."""
    learning_rate: float = 0.001
    """Learning rate."""

    scheduler_patience: int = 100
    """Number of epochs with no improvement before reducing the learning rate."""
    scheduler_factor: float = 0.8
    """Factor by which the learning rate is reduced on plateau."""

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
    per_structure_targets: list[str] = []
    """Targets to calculate per-structure losses."""
    log_mae: bool = False
    """Log MAE alongside RMSE."""
    log_separate_blocks: bool = False
    """Log per-block error."""
    best_model_metric: Literal["rmse_prod", "mae_prod", "loss"] = "rmse_prod"
    """Metric used to select best checkpoint (e.g., ``rmse_prod``)."""

    loss: str | dict[str, LossSpecification] = "mse"
    """This section describes the loss function to be used. See the
    :ref:`loss-functions` for more details."""
