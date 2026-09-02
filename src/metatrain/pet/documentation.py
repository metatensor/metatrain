"""
PET
===

PET is a cleaner, more user-friendly reimplementation of the original
PET model :footcite:p:`pozdnyakov_smooth_2023`. It is designed for better
modularity and maintainability, while preseving compatibility with the original
PET implementation in ``metatrain``. It also adds new features like long-range
features, better fine-tuning implementation, a possibility to train on
arbitrarty targets, and a faster inference due to the ``fast attention``.

{{SECTION_INSTALLATION}}

Additional outputs
------------------

In addition to the targets defined in the dataset, the PET architecture can also output
the following additional quantity:

- ``feature``: the internal PET features, before the different heads for each target.
- :ref:`mtt-aux-target-last-layer-features`: The features for a given target, taken
  before the last linear layer of the corresponding head.

Charge and spin conditioning
----------------------------

PET can condition its predictions on the total charge and spin multiplicity of
each system, so that the same structure can yield different predictions for
different electronic states (e.g. a neutral singlet vs. a charged doublet).
Enable it with the ``system_conditioning`` model hyperparameter and provide the
per-system values as ``extra_data`` in the training options file:

.. code-block:: yaml

    architecture:
      name: pet
      model:
        system_conditioning: true
        # optional, these are the defaults:
        max_charge: 10              # supports charges in [-10, 10]
        max_spin_multiplicity: 10   # supports 2S+1 values in [1, 10]

    training_set:
      systems:
        read_from: dataset.xyz
        length_unit: angstrom
      targets:
        energy:
          key: energy
          unit: eV
      extra_data:
        charge:
          key: charge                # from atoms.info["charge"]
        spin_multiplicity:
          key: spin_multiplicity     # from atoms.info["spin_multiplicity"]

The same ``extra_data`` section is used for evaluation with ``mtt eval``:

.. code-block:: yaml

    systems:
      read_from: test.xyz
    targets:
      energy:
        key: energy
        unit: eV
    extra_data:
      charge:
        key: charge
      spin_multiplicity:
        key: spin_multiplicity

The examples above read the values from ``atoms.info`` of an ASE-readable file.
The same ``extra_data`` section can also be used with zip and memory-mapped
datasets. In zip datasets, charge and spin conditioning are read from
``charge.mts`` and ``spin_multiplicity.mts``. In memory-mapped datasets, they
are read from ``charge.bin`` and ``spin_multiplicity.bin``. See
:ref:`dataset-formats` for details.

Systems without a value fall back to ``charge=0`` and ``spin_multiplicity=1``,
so a conditioned model can still be used on data without this information.
Values must be integers within ``[-max_charge, max_charge]`` and
``[1, max_spin_multiplicity]``; out-of-range or non-integer values raise an
error.

When running an exported model through the ASE calculator, the values are read
from ``atoms.info``:

.. code-block:: python

    from metatomic_ase import MetatomicCalculator

    atoms.info["charge"] = 1
    atoms.info["spin"] = 2  # spin multiplicity (2S + 1)
    atoms.calc = MetatomicCalculator("model.pt")
    energy = atoms.get_potential_energy()

{{SECTION_DEFAULT_HYPERS}}

Tuning hyperparameters
----------------------

The default hyperparameters above will work well in most cases, but they may not be
optimal for your specific dataset. There is good number of parameters to tune, both for
the :ref:`model <arch-{{architecture}}_model_hypers>` and the :ref:`trainer
<arch-{{architecture}}_trainer_hypers>`. Since seeing them for the first time might be
overwhelming, here we provide a **list of the parameters that are in general the most
important** (in decreasing order of importance):

.. container:: mtt-hypers-remove-classname

  .. autoattribute:: {{model_hypers_path}}.cutoff
      :no-index:

  .. autoattribute:: {{model_hypers_path}}.num_neighbors_adaptive
      :no-index:

  .. autoattribute:: {{trainer_hypers_path}}.learning_rate
      :no-index:

  .. autoattribute:: {{trainer_hypers_path}}.batch_size
      :no-index:

  .. autoattribute:: {{model_hypers_path}}.d_pet
      :no-index:

  .. autoattribute:: {{model_hypers_path}}.d_node
      :no-index:

  .. autoattribute:: {{model_hypers_path}}.num_gnn_layers
      :no-index:

  .. autoattribute:: {{model_hypers_path}}.num_attention_layers
      :no-index:

  .. autoattribute:: {{trainer_hypers_path}}.loss
      :no-index:

  .. autoattribute:: {{model_hypers_path}}.long_range
      :no-index:
"""

from typing import Any, Dict, Literal, Optional, Union

from typing_extensions import NotRequired, TypedDict

from metatrain.composition.documentation import FixedCompositionWeights
from metatrain.scaler.documentation import FixedScalerWeights
from metatrain.utils.ensemble import ShallowEnsembleHypersField
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.long_range import LongRangeHypers
from metatrain.utils.loss import LossSpecification

from .modules.finetuning import FinetuneHypers, NoFinetuneHypers


class ModelHypers(TypedDict):
    """Hyperparameters for the PET model."""

    cutoff: float = 4.5
    """Cutoff radius for neighbor search.

    This should be set to a value after which most of the interactions
    between atoms is expected to be negligible. A lower cutoff will lead
    to faster models.
    """
    num_neighbors_adaptive: Optional[int] = None
    """Target number of neighbors for the adaptive cutoff scheme.

    This parameter activates the adaptive cutoff functionality.
    Each atomic environments has a different cutoff, that is chosen
    such that the number of neighbors is approximately equal to this
    value. This can be useful to have a more uniform number of neighbors
    per atom, especially in sparse systems. Setting it to None disables
    this feature and uses all neighbors within the fixed cutoff radius.
    """
    adaptive_cutoff_method: Literal["grid", "solver"] = "solver"
    """Algorithm used to compute the per-atom adaptive cutoffs.

    ``"grid"`` evaluates the smoothed neighbor count on a discrete probe-cutoff
    grid and returns a Gaussian-weighted average of the probes (legacy
    behaviour). ``"solver"`` solves ``n_total(r) = num_neighbors_adaptive`` via
    a Newton-bisection root finder (default; faster and more accurate). Only
    has effect when ``num_neighbors_adaptive`` is set.
    """
    cutoff_function: Literal["Cosine", "Bump"] = "Bump"
    """Type of the smoothing function at the cutoff"""
    cutoff_width: float = 0.5
    """Width of the smoothing function at the cutoff"""
    cutoff_width_adaptive: float = 1.0
    """Width of the smooth cutoff taper used by the adaptive cutoff scheme.

    This controls the taper width of the smoothed neighbor count used to
    compute the per-atom adaptive cutoffs. Only has effect when
    ``num_neighbors_adaptive`` is set.
    """
    d_pet: int = 128
    """Dimension of the edge features.

    This hyperparameters controls width of the neural network. In general,
    increasing it might lead to better accuracy, especially on larger datasets, at the
    cost of increased training and evaluation time.
    """
    d_head: Union[int, Dict[str, int]] = 128
    """Output dimension of the node/edge heads.

    Either a single ``int`` (the same head dimension for the node and the edge
    heads) or a dict ``{node: int, edge: int}`` setting them independently. See
    :attr:`head_type` and :attr:`num_head_layers`.
    """
    head_type: Union[
        Literal["per_target", "per_block"],
        Dict[str, Literal["per_target", "per_block"]],
    ] = "per_target"
    """How the node/edge heads are shared across a target's blocks.

    ``"per_target"`` (default): one node head and one edge head per target,
    shared across all of that target's blocks. This is the standard PET
    behaviour.

    ``"per_block"``: a separate node/edge head per block of each target, so that
    every block can filter the backbone features for its own symmetry
    independently. Note that this multiplies the cost of the (nonlinear) heads by
    the number of blocks; for the edge heads, which act on the
    ``(n_atoms, max_neighbors)`` array, that cost is usually the dominant term of
    the whole model.

    May also be set *per target* by passing a dict keyed by target name (like the
    ``loss`` option), e.g. ``{"mtt::foo": "per_block"}``. A bare value applies to
    all targets; with a dict, targets not listed fall back to ``"per_target"``.
    """
    num_head_layers: int = 2
    """Number of Linear+SiLU layers in each node/edge head. Must be ``>= 1``.

    Each head maps ``d_node`` -> ``d_head`` (nodes) or ``d_pet`` -> ``d_head``
    (edges) through ``num_head_layers`` linear layers with SiLU activations,
    before the (linear) readout. See :attr:`head_type` and :attr:`d_head`.
    """
    d_node: int = 256
    """Dimension of the node features.

    Increasing this hyperparameter might lead to better accuracy,
    with a relatively small increase in inference time.
    """
    d_feedforward: int = 256
    """Dimension of the feedforward network in the attention layer."""
    num_heads: int = 8
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
    normalization: Literal["RMSNorm", "LayerNorm"] = "RMSNorm"
    """Layer normalization type."""
    activation: Literal["SiLU", "SwiGLU"] = "SwiGLU"
    """Activation function."""
    attention_temperature: float = 1.0
    """The temperature scaling factor for attention scores."""
    transformer_type: Literal["PreLN", "PostLN"] = "PreLN"
    """The order in which the layer normalization and attention
    are applied in a transformer block. Available options are ``PreLN``
    (normalization before attention) and ``PostLN`` (normalization after attention)."""
    geometry_embedding_l_max: Optional[int] = None
    """Maximum angular order of an alternative edge geometry embedding, based on
    normalized regular spherical harmonics scaled by the edge length. Each
    ``ell`` block of the harmonics is normalized to have unit norm, so that after
    scaling by the edge length ``|r|``, each block's norm equals ``|r|``.

    If ``None`` (default), the standard PET embedding of the raw edge vector and
    distance is used instead."""
    featurizer_type: Literal["residual", "feedforward"] = "feedforward"
    """Implementation of the featurizer of the model to use. Available
    options are ``residual`` (the original featurizer from the PET paper, that uses
    residual connections at each GNN layer for readout) and ``feedforward`` (a modern
    version that uses the last representation after all GNN iterations for readout).
    Additionally, the feedforward version uses bidirectional features flow during the
    message passing iterations, that favors features flowing from atom ``i`` to atom
    ``j`` to be not equal to the features flowing from atom ``j`` to atom ``i``."""
    readout_type: Dict[str, Any] = {"atom_type_gating": False, "hypers": {}}
    """Atom-type conditioning of the (linear) readout, i.e. the last layers.

    The readout is a strictly *linear* map from the head dimension to each block's
    output dimension; all nonlinearity lives in the heads (see :attr:`head_type`).
    This hyper controls optional atom-type conditioning of that linear map.

    ``{atom_type_gating: false}`` (default): a single shared linear readout per
    block, with no atom-type conditioning. This is the standard PET readout.

    ``{atom_type_gating: "one-hot"}``: an independent linear readout per atomic
    type, selected by the central-atom type. This is the natural readout for
    targets whose blocks span several atomic types, such as atom-centered basis
    expansions of a scalar field, where each type needs its own map onto a
    property axis padded to the largest per-type basis.

    ``{atom_type_gating: "moe", hypers: {...}}``: a mixture-of-experts linear
    readout whose experts are gated by routing weights from a learned embedding of
    the central-atom type. Note that every expert is evaluated for every atom, so
    this costs ``num_experts`` times a plain readout.

    .. code-block:: yaml

        readout_type:
          atom_type_gating: moe
          hypers:
            num_experts: 5
            num_routed_experts: 5
            num_topk_experts: 2
            embedding_dim: 16   # optional, default 16

    May also be set *per target* by passing a dict keyed by target name whose
    values are per-target specs. A spec containing the ``atom_type_gating`` key
    applies to all targets; otherwise the dict is read as ``{target_name: spec}``
    and targets not listed fall back to the default (no conditioning):

    .. code-block:: yaml

        readout_type:
          mtt::foo:
            atom_type_gating: one-hot
    """
    shallow_ensemble: Optional[ShallowEnsembleHypersField] = None
    """Optional shallow ensemble of the heads and/or readouts, for cheap
    uncertainty quantification and (with the ``ensemble_nll`` loss below)
    reduced equivariance error.

    When set, ``members`` independent copies of the head and/or readout modules
    (see :attr:`scope <metatrain.utils.ensemble.ShallowEnsembleHypers.scope>`
    below) are trained jointly on top of the shared backbone. The primary
    target output is the mean over members; the variance over members is
    exposed as an auxiliary ``{target}_uncertainty`` output (e.g.
    ``mtt::aux::energy_uncertainty``, or bare ``energy_uncertainty`` for the
    energy target). Two ways to train the members are supported, chosen via the
    target's loss (in the trainer hyperparameters):

    - a plain loss (``mse``, ``mae``, ...): computed on the mean, exactly as
      without ensembling. Members only diverge through independent random
      initialization (and, if enabled, ``dropout``/``bagging`` below).
    - ``ensemble_nll``: a Gaussian negative log-likelihood scoring the mean
      against the target using the ensemble variance as the predictive
      variance (``0.5 * ((mean - target)^2 / var + log(var))``). Actively
      encourages the members' spread to track the actual error.

    On a shared backbone, independent initialization alone is often not enough
    diversity between members to reliably beat a single model on accuracy --
    ``dropout`` and ``bagging`` are two independent, optional ways to inject
    more, at no extra inference cost (both are training-only).

    ``None`` (default) disables ensembling: the model has a single (non-ensembled)
    head/readout per target, exactly as without this hyperparameter.

    .. code-block:: yaml

        shallow_ensemble:
          scope: head      # or "readout"; see ShallowEnsembleHypers.scope
          members: 4       # must be > 1
          dropout: 0.1     # optional, scope="head" only; default 0 (off)
          bagging: 0.8     # optional; default 1.0 (off)

    See :class:`metatrain.utils.ensemble.ShallowEnsembleHypers` for the full
    description of ``scope``, ``members``, ``dropout`` and ``bagging``.
    """
    zbl: bool = False
    """Use ZBL potential for short-range repulsion"""
    long_range: LongRangeHypers = init_with_defaults(LongRangeHypers)
    """Long-range Coulomb interactions parameters."""
    system_conditioning: bool = False
    """Enable charge and spin conditioning embeddings. When enabled, per-system
    charge and spin multiplicity are embedded and added to node features at each
    GNN layer, allowing different predictions for the same structure under
    different electronic states."""
    max_charge: int = 10
    """Maximum absolute charge for the conditioning embedding table. Supports
    charges in the range ``[-max_charge, +max_charge]``."""
    max_spin_multiplicity: int = 10
    """Maximum spin multiplicity (2S+1) for the conditioning embedding table.
    Supports values in the range ``[1, max_spin_multiplicity]``."""


class TrainerHypers(TypedDict):
    """Hyperparameters for training PET models."""

    distributed: NotRequired[bool]
    """Whether to use distributed training. When not set, distributed training
    is enabled automatically when running under more than one SLURM task.
    Setting this option explicitly is deprecated."""
    distributed_port: int = 39591
    """Port for distributed communication among processes"""
    batch_size: int = 16
    """The number of samples to use in each batch of training. This
    hyperparameter controls the tradeoff between training speed and memory usage. In
    general, larger batch sizes will lead to faster training, but might require more
    memory."""
    num_epochs: int = 1000
    """Number of epochs."""
    warmup_fraction: float = 0.01
    """Fraction of training steps used for learning rate warmup."""
    learning_rate: float = 1e-4
    """Learning rate."""
    weight_decay: Optional[float] = None

    log_interval: int = 1
    """Interval to log metrics."""
    checkpoint_interval: int = 100
    """Interval to save checkpoints."""
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
    fixed_scaling_weights: FixedScalerWeights | str = {}
    """Weights for target scaling.

    This is passed to the ``fixed_weights`` argument of
    :meth:`Scaler.train_model <metatrain.scaler.Scaler.train_model>`,
    see its documentation to understand exactly what to pass here.

    Apart from those options, one can pass a path to a model checkpoint. If that
    is the checkpoint of a Scaler model, the pre-trained scaler will be loaded.
    When passing a checkpoint for the scaler, ``atomic_baseline`` must also
    be a checkpoint for a composition model.
    """
    per_structure_targets: list[str] = []
    """Targets to calculate per-structure losses and errors on."""
    num_workers: Optional[int] = None
    """Number of workers for data loading. If not provided, it is set
    automatically."""
    log_mae: bool = True
    """Log MAE alongside RMSE"""
    log_separate_blocks: bool = False
    """Log per-block error."""
    best_model_metric: Literal["rmse_prod", "mae_prod", "loss"] = "mae_prod"
    """Metric used to select best checkpoint (e.g., ``rmse_prod``)"""
    grad_clip_norm: float = 1.0
    """Maximum gradient norm value."""
    loss: str | dict[str, LossSpecification | str] = "mse"
    """This section describes the loss function to be used. See the
    :ref:`loss-functions` for more details."""
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

    finetune: NoFinetuneHypers | FinetuneHypers = {
        "read_from": None,
        "method": "full",
        "config": {},
        "inherit_heads": {},
    }
    """Parameters for fine-tuning trained PET models.

    See :ref:`label_fine_tuning_concept` for more details.
    """
