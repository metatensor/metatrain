"""
LOREM (Experimental)
====================

**LOREM** (*Learning Long-Range Representations with Equivariant Messages*)
:footcite:p:`lorem_2025` predicts energies (forces and stress by autograd) and
per-atom Born effective charges. Short-range features are a spherical
neighbor density with species-dependent radial filters. Long-range features
are Coulomb potentials of learned charges, evaluated with
`torch-pme <https://github.com/lab-cosmo/torch-pme>`_.

Born effective charges need ``max_degree >= 2``.

{{SECTION_INSTALLATION}}

{{SECTION_DEFAULT_HYPERS}}

Tuning hyperparameters
----------------------

The most impactful hyperparameters (roughly in decreasing order of importance):

.. container:: mtt-hypers-remove-classname

  .. autoattribute:: {{model_hypers_path}}.cutoff
      :no-index:

  .. autoattribute:: {{model_hypers_path}}.num_features
      :no-index:

  .. autoattribute:: {{model_hypers_path}}.max_degree
      :no-index:

  .. autoattribute:: {{trainer_hypers_path}}.learning_rate
      :no-index:

Default values follow the LOREM paper (5 Å cutoff, 128 features). Reduce
``max_degree``, ``num_radial`` and ``num_features`` for faster iteration.

{{SECTION_MODEL_HYPERS}}

"""

from typing import Literal, Optional

from typing_extensions import NotRequired, TypedDict

from metatrain.composition.documentation import FixedCompositionWeights
from metatrain.scaler.documentation import FixedScalerWeights
from metatrain.utils.loss import LossSpecification


###########################
#  MODEL HYPERPARAMETERS  #
###########################


class ModelHypers(TypedDict):
    """Hyperparameters for the experimental LOREM model."""

    cutoff: float = 5.0
    """Short-range cutoff radius, in the length unit of the dataset.

    The cosine cutoff goes to zero at this radius. Ewald smearing is
    ``cutoff / 4`` and the reciprocal-space wavelength is ``cutoff / 8``.
    """
    max_degree: int = 6
    """Maximum angular momentum of the short-range spherical features.
    Born effective charges need at least 2."""
    max_degree_lr: int = 2
    """Maximum angular momentum of the long-range charges. Must not exceed
    ``max_degree``. The paper default is 2."""
    num_features: int = 128
    """Width of the scalar node features."""
    num_spherical_features: int = 8
    """Width of the spherical node features."""
    num_radial: int = 32
    """Number of Bernstein radial basis functions."""
    num_species: int = 8
    """Width of the learned species embedding."""
    num_message_passing: int = 0
    """Extra message-passing steps after the initial density. The paper
    default is 0."""
    equivariant_message_passing: bool = True
    """When message passing is on, also update the spherical features.
    Ignored when ``num_message_passing`` is 0."""
    initialize_node_features: bool = True
    """Start the scalar node features from a projection of the species
    embedding. When false they start from zero, as in lorem-jax's ``LoremBEC``."""


##############################
#  TRAINER HYPERPARAMETERS   #
##############################


class TrainerHypers(TypedDict):
    """Hyperparameters for training LOREM models."""

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
    warmup_fraction: float = 0.01
    """Fraction of training steps used for learning rate warmup, before the
    cosine decay."""
    learning_rate: float = 0.001
    """Learning rate."""
    weight_decay: Optional[float] = None
    """Weight decay. If set, AdamW is used instead of Adam."""

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
    atomic_baseline: FixedCompositionWeights | str = {}
    """The baselines for each target.

    By default, ``metatrain`` fits a linear model (:class:`CompositionModel
    <metatrain.composition.CompositionModel>`) to compute the least-squares
    baseline for each atomic species for each target.

    This hyperparameter instead lets you provide those baselines, either as a
    dictionary or as a path to a pre-trained composition model checkpoint:

    - a dictionary whose keys are target names, and whose values are either
      one baseline for every atomic type, or a dictionary mapping atomic types
      to baselines.
    - a string path to a ``.ckpt`` file from a pre-trained composition model.

    For example:

    - ``atomic_baseline: {"energy": {1: -0.5, 6: -10.0}}`` fixes the energy
      baseline for hydrogen (Z=1) to -0.5 and for carbon (Z=6) to -10.0, and
      fits the rest.
    - ``atomic_baseline: {"energy": -5.0}`` fixes the energy baseline for
      every atomic type to -5.0.
    - ``atomic_baseline: {"mtt:dos": 0.0}`` sets that target's baseline to
      0.0, which disables the atomic baseline for it.
    - ``atomic_baseline: "/path/to/model.ckpt"`` loads a pre-trained
      composition model checkpoint instead of fitting one.

    The baseline is subtracted from the targets during training and added
    back at evaluation. It is a per-atom contribution, so for a per-structure
    target such as the total energy it is multiplied by the number of atoms
    of that type.
    """
    fixed_scaling_weights: FixedScalerWeights | str = {}
    """Weights for target scaling.

    This is passed to the ``fixed_weights`` argument of
    :meth:`Scaler.train_model <metatrain.scaler.Scaler.train_model>`,
    see its documentation to understand exactly what to pass here.

    A path to a model checkpoint is also accepted. If that checkpoint is a
    Scaler model, the pre-trained scaler is loaded. When passing a checkpoint
    for the scaler, ``atomic_baseline`` must also be a checkpoint for a
    composition model.
    """
    per_structure_targets: list[str] = []
    """Targets to calculate per-structure losses."""
    num_workers: Optional[int] = None
    """Number of workers for data loading. If not provided, it is set
    automatically."""
    log_mae: bool = False
    """Log MAE alongside RMSE."""
    log_separate_blocks: bool = False
    """Log per-block error."""
    best_model_metric: Literal["rmse_prod", "mae_prod", "loss"] = "rmse_prod"
    """Metric used to select best checkpoint (e.g., ``rmse_prod``)."""
    grad_clip_norm: float = 1.0
    """Maximum gradient norm value."""

    loss: str | dict[str, LossSpecification | str] = "mse"
    """This section describes the loss function to be used. See the
    :ref:`loss-functions` for more details."""
