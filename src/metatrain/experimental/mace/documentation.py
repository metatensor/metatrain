"""
MACE
====

Higher order equivariant message passing graph neural network
:footcite:p:`batatia2022mace`.

.. _architecture-mace_implementation:

Implementation
--------------

The architecture is a very thin wrapper around the official MACE implementation, which
is `hosted here <https://github.com/ACEsuit/mace>`_. Arbitrary heads are added on top of
MACE to predict an arbitrary number of targets with arbitrary symmetry properties. These
heads take as input the output node features of MACE and pass them through a linear
layer (or MLP for invariant targets) to obtain the final predictions.

One important feature is that the architecture is ready to take a pretrained MACE model
file as input. The heads required to predict the targets will be added on top of the
MACE model, so one can continue training for arbitrary targets. See the
``mace_model`` hyperparameter for more details. For simply exporting a foundation MACE
model to use as a ``metatomic`` model (e.g. in ASE or LAMMPS), see :ref:`exporting a
foundation MACE model <architecture-mace_export_foundation_model>`.

{{SECTION_INSTALLATION}}

{{SECTION_DEFAULT_HYPERS}}

Tuning hyperparameters
----------------------

The default hyperparameters above will work well in most cases, but they
may not be optimal for your specific dataset. There is good number of
parameters to tune, both for the
:ref:`model <architecture-{{architecture}}_model_hypers>` and the
:ref:`trainer <architecture-{{architecture}}_trainer_hypers>`. Since seeing them
for the first time might be overwhelming, here we provide a **list of the
parameters that are in general the most important**:

.. container:: mtt-hypers-remove-classname

  .. autoattribute:: {{model_hypers_path}}.mace_model
      :no-index:

  .. autoattribute:: {{model_hypers_path}}.r_max
      :no-index:

  .. autoattribute:: {{trainer_hypers_path}}.learning_rate
      :no-index:

  .. autoattribute:: {{trainer_hypers_path}}.batch_size
      :no-index:

  .. autoattribute:: {{model_hypers_path}}.hidden_irreps
      :no-index:

  .. autoattribute:: {{model_hypers_path}}.correlation
      :no-index:

  .. autoattribute:: {{model_hypers_path}}.num_interactions
      :no-index:

  .. autoattribute:: {{trainer_hypers_path}}.loss
      :no-index:

{{SECTION_MODEL_HYPERS}}

{{SECTION_TRAINER_HYPERS}}

.. _architecture-mace_export_foundation_model:

Exporting a foundation MACE model
---------------------------------

As it is now, exporting a foundation MACE model from one of their `provided model
files <https://github.com/ACEsuit/mace-foundations>`_ involves using ``mtt train``
with 0 epochs. To do so, use the following ``options.yaml`` file:

.. code-block:: yaml

    architecture:
        name: experimental.mace
        model:
            # Replace mace_model with the path to your file
            mace_model: path/to/foundation/mace/model.model
            mace_head_target: energy
        training:
            num_epochs: 0
            batch_size: 1

    training_set: dummy_dataset.xyz
    validation_set: dummy_dataset.xyz

with ``dummy_dataset.xyz`` being any dataset containing at least one structure
with just the ``energy`` property. For example, you can use:

.. code-block::

    2
    Properties=species:S:1:pos:R:3:forces:R:3 energy=-2.1
    H 0.0 0.0 0.0 0.0 0.0 0.0
    H 1.0 0.0 0.0 0.0 0.0 0.0

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
    mace_head_target: str = "energy"
    """Target to which the MACE head is related.

    ``metatrain`` adds arbitrary heads on top of MACE to predict arbitrary targets.
    However, MACE models have themselves a head. This hyperparameter specifies which
    metatrain target corresponds to the MACE head. For this target, no new head will be
    added, and the output of MACE's head will be used directly.

    .. note::
        MACE models with multiple heads also exist, but ``metatrain`` only supports
        using this hyperparameter to deal with single-head MACE models for now.
    """
    r_max: float = 5.0
    """Cutoff radius for neighbor search.

    This should be set to a value after which most of the interactions between atoms is
    expected to be negligible. A lower cutoff will lead to faster models.
    """
    num_radial_basis: int = 8
    """Number of radial basis functions for the radial embedding."""
    radial_type: Literal["bessel", "gaussian", "chebyshev"] = "bessel"
    """Type of radial basis functions to use in the radial embedding."""
    num_cutoff_basis: int = 5
    """Number of basis functions for smooth cutoff"""
    max_ell: int = 3
    r"""Highest :math:`\ell` of spherical harmonics used in the interactions.

    Note that this is not the maximum :math:`\ell` in ``hidden_irreps``, since
    hidden_irreps can contain :math:`\ell` values as high as ``max_ell*correlation``.
    """
    interaction: Literal[
        "RealAgnosticResidualInteractionBlock",
        "RealAgnosticAttResidualInteractionBlock",
        "RealAgnosticInteractionBlock",
        "RealAgnosticDensityInteractionBlock",
        "RealAgnosticDensityResidualInteractionBlock",
        "RealAgnosticResidualNonLinearInteractionBlock",
    ] = "RealAgnosticResidualInteractionBlock"
    """Name of interaction block.

    Class that will be used to compute interactions between atoms at each layer.
    """
    num_interactions: int = 2
    """Number of message passing steps.

    MACE's last message passing step only outputs scalar features, so if you are
    training on a target that is not scalar (e.g. a vector or some spherical
    tensor with higher order), the effective number of message passing steps for
    that target will be ``num_interactions - 1``.
    """
    hidden_irreps: str = "128x0e + 128x1o + 128x2e"
    r"""Irreps for hidden node features.

    This defines the shape of the node features at each layer of the MACE model
    (except for the last layer, which only contains scalars). The notation for
    the irreps is `e3nn's standard notation <https://docs.e3nn.org/en/stable/api/
    o3/o3_irreps.html>`_. Essentially, the irreps string is a sum of terms of the
    form ``{multiplicity}x{ell}{parity}``, where ``{multiplicity}`` is the number of
    channels with angular momentum ``{ell}`` and parity ``{parity}`` (``e`` for even,
    ``o`` for odd). For example, ``16x0e + 32x1o`` means that there are 16 scalar
    channels (:math:`\ell=0`) and 32 vector channels (:math:`\ell=1`) at each layer.

    Increasing the multiplicities makes the network wider, which generally leads to
    better accuracy at the cost of increased training and evaluation time.

    Increasing the maximum :math:`\ell` included in the irreps allows the network to
    capture more complex angular dependencies. However, its effect might be heavily
    dependent on your dataset and target. The hidden irreps should include at least
    up to the maximum :math:`\ell` of the target you are training on. For example, if
    you are training on dipole moments (:math:`\ell=1`), the hidden irreps should
    include at least :math:`\ell=1` channels.

    .. note::

        At the time of writing, MACE enforces that all channels of ``hidden_irreps``
        should have the same multiplicity.

    """
    edge_irreps: Optional[str] = None
    """Irreps for edge features."""
    apply_cutoff: bool = True
    """Apply cutoff to the radial basis functions before MLP"""
    avg_num_neighbors: float = 1
    """Normalization factor for the messages."""
    pair_repulsion: bool = False
    """Use pair repulsion term with ZBL potential"""
    distance_transform: Optional[Literal["Agnesi", "Soft"]] = None
    """Use distance transform for radial basis functions"""
    correlation: int = 3
    """Correlation order at each layer.

    After computing pair-wise (2-body) messages between atoms, MACE applies
    products that construct higher order correlations between messages.
    This hyperparameter controls the amount of products applied. For example,
    ``correlation=1`` means that the interactions are purely 2-body, while
    ``correlation=2`` would roughly equate to including 3-body interactions,
    and so on.

    This hyperparameter together with ``max_ell`` determine the maximum
    angular momentum that will be non-zero in ``hidden_irreps``, which is
    ``max_ell * correlation``.
    """
    gate: Optional[Literal["silu", "tanh", "abs"]] = "silu"
    """Non linearity used for the non linear readouts.

    This determines which kind of non-linearity is applied in the non linear
    readouts. The non linear readouts are: MACE's internal MLP readout (applied
    only at the last layer) and arbitrary MLP heads added on top of MACE by
    ``metatrain``.

    The non-linearity is applied only to scalar channels, therefore it won't
    have any effect for non-scalar targets.
    """
    interaction_first: Literal[
        "RealAgnosticResidualInteractionBlock",
        "RealAgnosticInteractionBlock",
        "RealAgnosticDensityInteractionBlock",
        "RealAgnosticDensityResidualInteractionBlock",
        "RealAgnosticResidualNonLinearInteractionBlock",
    ] = "RealAgnosticResidualInteractionBlock"
    """Name of interaction block for the first interaction layer.

    Class that will be used to compute interactions between atoms at the
    first layer.
    """
    MLP_irreps: str = "16x0e"
    """Hidden irreps of the MLP readouts.

    The MLP readouts are: MACE's internal MLP readout (applied
    only at the last layer) and arbitrary MLP heads added on top of MACE by
    ``metatrain``.

    The non-linearity is applied only to scalar channels, therefore these
    irreps should only contain scalar channels.
    """
    radial_MLP: list[int] = [64, 64, 64]
    """Width of the radial MLP.

    Only used for MACE's internal MLP.
    """
    use_embedding_readout: bool = False
    """Use embedding readout for the final output"""
    use_last_readout_only: bool = False
    """Use only the last readout for the final output.

    This is only used by the internal MACE readout, arbitrary heads by
    ``metatrain`` always use as input a concatenation of the node
    features from all layers.
    """
    use_agnostic_product: bool = False
    """Use element agnostic product"""


class TrainerHypers(TypedDict):
    # Optimizer hypers (directly using MACE's scripts)
    optimizer: Literal["adam", "adamw", "schedulefree"] = "adam"
    """Optimizer for parameter optimization"""
    learning_rate: float = 0.01  # Named "lr" in MACE
    """Learning rate of the optimizer."""
    weight_decay: float = 5e-07
    """Weight decay (L2 penalty)."""
    amsgrad: bool = True
    """Use amsgrad variant of optimizer."""
    beta: float = 0.9
    """Beta parameter for the optimizer"""

    # Scheduler hypers (directly using MACE's scripts)
    lr_scheduler: str = "ReduceLROnPlateau"  # Named "scheduler" in MACE
    """Type of learning rate scheduler."""
    lr_scheduler_gamma: float = 0.9993
    """Gamma parameter for learning rate scheduler."""
    lr_factor: float = 0.8
    """Learning rate factor"""
    lr_scheduler_patience: int = 50  # Named "scheduler_patience" in MACE
    """Scheduler patience."""

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
    atomic_baseline: FixedCompositionWeights = {}
    """The baselines for each target.

    By default, ``metatrain`` will fit a linear model (:class:`CompositionModel
    <metatrain.utils.additive.composition.CompositionModel>`) to compute the
    least squares baseline for each atomic species for each target.

    However, this hyperparameter allows you to provide your own baselines.
    The value of the hyperparameter should be a dictionary where the keys are the
    target names, and the values are either (1) a single baseline to be used for
    all atomic types, or (2) a dictionary mapping atomic types to their baselines.
    For example:

    - ``atomic_baseline: {"energy": {1: -0.5, 6: -10.0}}`` will fix the energy
      baseline for hydrogen (Z=1) to -0.5 and for carbon (Z=6) to -10.0, while
      fitting the baselines for the energy of all other atomic types, as well
      as fitting the baselines for all other targets.
    - ``atomic_baseline: {"energy": -5.0}`` will fix the energy baseline for
      all atomic types to -5.0.
    - ``atomic_baseline: {"mtt:dos": 0.0}`` sets the baseline for the "mtt:dos"
      target to 0.0, effectively disabling the atomic baseline for that target.

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

    .. note::
        If a MACE model is loaded through the ``mace_model`` hyperparameter, the
        atomic baselines in the MACE model are used by default for the target
        indicated in ``mace_head_target``. If you want to override them, you need
        to set explicitly the baselines for that target in this hyperparameter.
    """
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
