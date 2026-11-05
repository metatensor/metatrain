"""
se_e2_a (Experimental)
======================

This is an interface to the smooth-edition two-body embedding with angular
information (``se_e2_a``, historically ``se_a``) of `deepmd-kit
<https://github.com/deepmodeling/deepmd-kit>`_. The descriptor was introduced
for Deep Potential models in :footcite:t:`se_e2_a_2018`. Each atom is described
by the coordinates of its neighbors inside a cutoff, passed through an
embedding network and reduced to a rotationally invariant vector. A fitting
network, one per chemical element, maps that vector to an atomic energy.

.. note::

   The ``type_map`` required by deepmd-kit is derived from the atomic numbers
   present in the dataset. It is not a user-facing hyperparameter. Neighbor
   counts in ``descriptor.sel`` follow that same type order.

{{SECTION_INSTALLATION}}

{{SECTION_DEFAULT_HYPERS}}

Tuning hyperparameters
----------------------

The hyperparameters that usually matter most are the cutoff, how many
neighbors of each element are kept, and the width of the two networks.

.. container:: mtt-hypers-remove-classname

  .. autoattribute:: {{model_hypers_path}}.descriptor
      :no-index:

  .. autoattribute:: {{trainer_hypers_path}}.learning_rate
      :no-index:

  .. autoattribute:: {{trainer_hypers_path}}.batch_size
      :no-index:

``descriptor.rcut`` is the interaction range. ``descriptor.sel`` is the number
of neighbors retained for each element; an integer uses that count for every
element, and a list must contain one entry per element in ``type_map`` order.
``descriptor.neuron`` and ``descriptor.axis_neuron`` set the embedding, with
``axis_neuron`` no larger than the last embedding width. ``fitting_net.neuron``
sets the fitting network. The atomic energy is an affine function of that
network's last hidden activation.

Loading a DeepMD checkpoint
---------------------------

``se_e2_a_model`` points at a deepmd-kit training checkpoint (the ``.pt`` file
written by training, not the frozen model produced by ``dp --pt freeze``).
Single-task checkpoints load directly. A multi-task checkpoint needs
``se_e2_a_model_branch``; leaving it unset raises an error that lists the
branches stored in the file.

.. code-block:: yaml

    architecture:
        name: experimental.se_e2_a
        model:
            se_e2_a_model: path/to/deepmd-checkpoint.pt
        training:
            num_epochs: 0

Last-layer features and LLPR
----------------------------

The model exposes one auxiliary output per target,
``mtt::aux::<target>_last_layer_features``. ``se_e2_a`` does not share its
fitting network across elements, so the raw last hidden activation is not yet
a single linear feature: each element has its own final weights and bias. The
exported feature places an atom's activation into that element's block and
appends the element one-hot. The atomic energy is then one affine function of
this vector, which is what :ref:`arch-llpr` reads. System-level targets sum
the same per-atom features. Requesting the energy and the features together
evaluates the descriptor twice, because deepmd-kit's embedding entry point
does not return the energy.

{{SECTION_MODEL_HYPERS}}

with the following definitions needed to fully understand some of the parameters:

.. autoclass:: {{architecture_path}}.documentation.DescriptorHypers
    :members:
    :undoc-members:

.. autoclass:: {{architecture_path}}.documentation.FittingNetHypers
    :members:
    :undoc-members:

"""

from typing import Literal, Optional, Union

from typing_extensions import NotRequired, TypedDict

from metatrain.composition.documentation import FixedCompositionWeights
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification


###########################
#  MODEL HYPERPARAMETERS  #
###########################


class DescriptorHypers(TypedDict):
    """``se_e2_a`` descriptor hyperparameters."""

    sel: Union[int, list[int]] = 46
    """Neighbors kept for each element.

    An integer uses the same count for every element in the dataset. A list
    must contain one positive integer per element, in the dataset's atomic
    number order (the order of deepmd-kit's ``type_map``)."""
    rcut: float = 6.0
    """Cutoff radius, in the dataset's length unit."""
    rcut_smth: float = 0.5
    """Distance at which the neighbor weight starts to decay to zero.

    Must be smaller than ``rcut``."""
    neuron: list[int] = [25, 50, 100]
    """Hidden widths of the embedding network. The last width is the
    embedding dimension."""
    axis_neuron: int = 16
    """Number of axis neurons. Must be at least 1 and at most the last
    entry of ``neuron``."""
    resnet_dt: bool = False
    """Use a residual time step inside the embedding network."""
    activation_function: str = "tanh"
    """Activation used by the embedding network. deepmd-kit names such as
    ``"tanh"``, ``"gelu"`` and ``"silu"`` are accepted."""
    precision: int = 32
    """Floating-point precision of the descriptor, ``32`` or ``64``.

    This is fixed when the module is built. ``module.to(dtype)`` does not
    change it, so it has to match ``base_precision``."""
    type_one_side: bool = True
    """Share the embedding of a pair across the two orderings of its
    elements. This is the standard ``se_e2_a`` setting."""
    set_davg_zero: bool = False
    """Skip the data-dependent shift of the environment matrix and keep it
    centered at zero."""
    env_protection: float = 0.0
    """Small offset that protects the environment matrix when a neighbor
    distance underflows."""
    exclude_types: list[list[int]] = []
    """Pairs of type indices whose interactions are dropped. Indices follow
    ``type_map`` order."""
    trainable: bool = True
    """Whether the embedding parameters receive gradients."""
    seed: Optional[int] = 1
    """Seed for the embedding-network initialization."""


class FittingNetHypers(TypedDict):
    """Fitting-network hyperparameters."""

    neuron: list[int] = [240, 240, 240]
    """Hidden widths of the fitting network. The last width is the size of
    the activation that, after the type gating described above, becomes the
    last-layer feature."""
    resnet_dt: bool = True
    """Use a residual time step in each hidden layer."""
    seed: Optional[int] = 1
    """Seed for the fitting-network initialization."""
    precision: int = 32
    """Floating-point precision of the fitting network, ``32`` or ``64``.

    It may differ from ``descriptor.precision``. The model's reported dtype
    follows the descriptor."""
    activation_function: str = "tanh"
    """Activation used by the fitting network."""
    type: str = "ener"
    """Fitting head. ``"ener"`` predicts an atomic energy."""
    numb_fparam: int = 0
    """Number of frame-level parameters concatenated to the descriptor."""
    numb_aparam: int = 0
    """Number of atom-level parameters concatenated to the descriptor."""
    dim_case_embd: int = 0
    """Width of an optional case embedding."""
    trainable: bool = True
    """Whether the fitting parameters receive gradients."""
    rcond: Optional[float] = None
    """Cutoff used when deepmd-kit solves for the atomic-energy bias."""
    use_aparam_as_mask: bool = False
    """Treat the atom-level parameters as a mask instead of extra inputs."""


class ModelHypers(TypedDict):
    """Hyperparameters for an ``se_e2_a`` model.

    The ``type_map`` needed by deepmd-kit is taken from the dataset's atomic
    numbers and is not set here.
    """

    se_e2_a_model: Optional[str] = None
    """Path to a deepmd-kit training checkpoint. A frozen model written by
    ``dp --pt freeze`` is rejected: the configuration needed to rebuild the
    module lives in the training checkpoint. When this is set, the weights
    are loaded from the file instead of being initialized from the
    hyperparameters below. Energy biases and standard deviations stored in
    the deepmd-kit model are handed to metatrain's composition model and
    scaler."""
    se_e2_a_model_branch: Optional[str] = None
    """Task name to load from a multi-task checkpoint. If the file contains
    several tasks and this is unset, the error lists those tasks."""
    descriptor: DescriptorHypers = init_with_defaults(DescriptorHypers)
    """Descriptor configuration."""
    fitting_net: FittingNetHypers = init_with_defaults(FittingNetHypers)
    """Fitting-network configuration."""


##############################
#  TRAINER HYPERPARAMETERS   #
##############################


class TrainerHypers(TypedDict):
    """Hyperparameters for training ``se_e2_a`` models."""

    distributed: NotRequired[bool]
    """Whether to use distributed training. When not set, distributed training
    is enabled automatically when running under more than one SLURM task.
    Setting this option explicitly is deprecated."""
    distributed_port: int = 39591
    """Port for DDP communication."""
    batch_size: int = 8
    """The number of samples to use in each batch of training. This
    hyperparameter controls the tradeoff between training speed and memory usage. In
    general, larger batch sizes will lead to faster training, but might require more
    memory."""
    max_atoms_per_batch: Optional[int] = None
    """If set, use greedy atom-count packing instead of fixed ``batch_size``.
    Structures are accumulated into each batch until adding another would exceed this
    limit, producing variable numbers of structures per batch. Supported with any
    dataset type. When set, ``batch_size`` is ignored for constructing training
    and validation batches (it is still used internally for composition model and
    scaler fitting)."""
    min_atoms_per_batch: int = 0
    """Minimum total number of atoms required to keep a batch when
    ``max_atoms_per_batch`` is set. Batches whose total atom count falls below this
    threshold are discarded during packing. Defaults to ``0`` (no minimum)."""
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
    """
    Normalize targets to unit std during training.

    If true, a single scale is computed for each target, given by the uncentered
    standard deviation across all values in the dataset for that target.

    For targets with more than one property (i.e. > 1 block or >= 1 block with > 1
    property), per-property scales are also computed, and used to re-scale model
    predictions.

    See also :ref:`scale-targets`.
    """
    fixed_composition_weights: FixedCompositionWeights = {}
    """Weights for atomic contributions.

    This is passed to the ``fixed_weights`` argument of
    :meth:`CompositionModel.train_model
    <metatrain.composition.CompositionModel.train_model>`,
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
