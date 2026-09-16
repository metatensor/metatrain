"""
LOREM (Experimental)
====================

A PyTorch / TorchScript implementation of **LOREM** (*Learning Long-Range
Representations with Equivariant Messages*) :footcite:p:`lorem_2025` for
``metatrain``.

LOREM combines a short-range spherical descriptor with long-range Coulomb
features computed from learned atomic charges. The official reference code is
the JAX package `lorem-jax <https://github.com/lab-cosmo/lorem-jax>`_; this
architecture is a metatrain-native port so that models can be trained with
``mtt`` and exported as ``metatomic`` TorchScript files.

The short-range block builds a spherical-harmonic neighbor density (Bessel
radial basis × real spherical harmonics), contracts it to rotationally
invariant features, and optionally applies a few scalar message-passing
updates. The long-range block maps scalar features to a charge channel and
spherical features to charges up to ``max_degree_lr``, then evaluates their
Coulomb potential in parallel with
`torch-pme <https://github.com/lab-cosmo/torch-pme>`_ (Ewald / P3M / direct).
That is the paper's equivariant long-range message: Ewald summation over each
:math:`(\\ell, m)` independently.

.. note::

   Scalar targets (typically energy, forces and stress via autograd),
   Cartesian rank-1 dipoles (PhysNet-style :math:`\\mu = \\sum_i q_i r_i`)
   and per-atom Cartesian rank-2 targets (Born effective charges / APT, the
   ``lorem.LoremBEC`` head) are supported. Spherical charges use the
   Clebsch-Gordan self-product (``TensorDense``, the ``e3x.nn.TensorDense``
   port). ``max_degree >= 2`` is required for BEC.

   This is a metatrain-native port: same equations and knobs as the paper /
   lorem-jax, not a bit-exact JAX clone. Developer comparison notes and the
   in-repo contract tests live in the architecture ``README.md``.

{{SECTION_INSTALLATION}}

{{SECTION_DEFAULT_HYPERS}}

Tuning hyperparameters
----------------------

The most impactful hyperparameters (roughly in decreasing order of importance):

.. container:: mtt-hypers-remove-classname

  .. autoattribute:: {{model_hypers_path}}.cutoff
      :no-index:

  .. autoattribute:: {{model_hypers_path}}.long_range
      :no-index:

  .. autoattribute:: {{model_hypers_path}}.num_features
      :no-index:

  .. autoattribute:: {{model_hypers_path}}.max_degree
      :no-index:

  .. autoattribute:: {{trainer_hypers_path}}.learning_rate
      :no-index:

Default values follow the LOREM paper / ``lorem-jax`` (5 Å cutoff, long-range
on, 128 features). Reduce ``max_degree``, ``num_radial`` and ``num_features``
for faster iteration.

{{SECTION_MODEL_HYPERS}}

"""

from typing import Literal, Optional

from typing_extensions import NotRequired, TypedDict

from metatrain.composition.documentation import FixedCompositionWeights
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification


###########################
#  MODEL HYPERPARAMETERS  #
###########################


class LoremLongRangeHypers(TypedDict):
    """Long-range Coulomb hyperparameters. LOREM's long-range Ewald message
    is a core part of the architecture (not an optional add-on like other
    architectures' ``long_range``), so it cannot be disabled."""

    use_ewald: bool = True
    """Use Ewald summation for periodic systems during training. If False,
    P3M is used instead."""
    smearing: float = 1.4
    """Smearing width in Fourier space."""
    kspace_resolution: float = 1.33
    """Resolution of the reciprocal space grid."""
    interpolation_nodes: int = 5
    """Number of grid points for interpolation (for PME only)."""
    lr_scale_init: float = 1.0
    """Initial value of the learnable ``lr_scale`` that multiplies the
    long-range residual. ``1.0`` trains the long-range branch from scratch;
    set ``0.0`` for a zero perturbation when warm-starting the short-range
    trunk."""


class ModelHypers(TypedDict):
    """Hyperparameters for the experimental LOREM model."""

    cutoff: float = 5.0
    """Short-range cutoff radius, in the length unit of the dataset."""
    cutoff_width: float = 0.5
    """Width of the cosine cutoff envelope. The envelope is 1 inside
    ``cutoff - cutoff_width`` and 0 at ``cutoff``."""
    max_degree: int = 6
    """Maximum angular momentum :math:`\\ell` of the short-range spherical
    density. Values above 2 require ``sphericart-torch``."""
    max_degree_lr: int = 2
    """Maximum angular momentum of long-range charges. Ewald / P3M / direct
    summation is run independently on each :math:`(\\ell, m)` channel, plus
    one extra scalar charge from the invariant features. Must not exceed
    ``max_degree``. The paper default is 2 (10 charge channels)."""
    num_features: int = 128
    """Dimension of invariant atom features and of the energy readout."""
    num_spherical_features: int = 8
    """Feature width of the Clebsch-Gordan ``TensorDense`` self-product on
    the short-range spherical density, and of the long-range charge /
    potential mix."""
    num_radial: int = 32
    """Number of radial basis functions (Bernstein or Bessel)."""
    radial_basis: Literal["bessel", "basic_bernstein"] = "basic_bernstein"
    """Radial expansion on neighbor distances. ``basic_bernstein`` is the
    e3x / lorem-jax default; ``bessel`` is the original sinc basis."""
    sh_convention: Literal["orthonormal", "e3x"] = "e3x"
    """Spherical-harmonic normalization. ``e3x`` is Racah / Schmidt
    (lorem-jax); ``orthonormal`` is 4π-normalized (sphericart). Both use
    ``m = -ℓ … +ℓ`` so Clebsch–Gordan is unchanged."""
    trunk: Literal["spherical", "pet"] = "spherical"
    """Short-range trunk. ``spherical`` is the paper descriptor.
    ``pet`` mounts metatrain's ``PETBackend`` as the short-range trunk and
    keeps a spherical sidecar for equivariant charges / BEC."""
    pet: dict = {}
    """Optional overrides merged into default PET model hypers when
    ``trunk`` is ``pet`` (``d_pet``, ``num_gnn_layers``, …). ``d_node``
    is always set to ``num_features``."""
    num_message_passing: int = 0
    """Number of additional scalar message-passing layers after the spherical
    density. The paper default is 0 (descriptor + long-range only)."""
    long_range: LoremLongRangeHypers = init_with_defaults(LoremLongRangeHypers)
    """Long-range Coulomb features from learned equivariant atomic charges."""


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
    learning_rate: float = 0.001
    """Learning rate."""

    scheduler: str = "plateau"
    """Learning-rate schedule: ``"plateau"`` (``ReduceLROnPlateau``, driven by
    ``scheduler_patience``/``scheduler_factor``) or ``"cosine"`` (linear warmup
    then cosine decay over the full run, driven by ``warmup_fraction`` -- the
    same schedule PET and SOAP-BPNN use, for apples-to-apples comparisons)."""
    scheduler_patience: int = 100
    """Number of epochs with no improvement before reducing the learning rate.
    Only used when ``scheduler`` is ``"plateau"``."""
    scheduler_factor: float = 0.8
    """Factor by which the learning rate is reduced on plateau. Only used when
    ``scheduler`` is ``"plateau"``."""
    warmup_fraction: float = 0.01
    """Fraction of total optimizer steps spent on linear warmup before the
    cosine decay begins. Only used when ``scheduler`` is ``"cosine"``."""

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
