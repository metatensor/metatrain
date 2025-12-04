"""
PhACE
=====

PhACE is a physics-inspired equivariant neural network architecture. Compared to, for
example, MACE and GRACE, it uses a geometrically motivated basis and a fast and
elegant tensor product implementation. The tensor product used in PhACE leverages a
equivariant representation that differs from the typical spherical one. You can read
more about it here: https://pubs.acs.org/doi/10.1021/acs.jpclett.4c02376.

{{SECTION_INSTALLATION}}

{{SECTION_DEFAULT_HYPERS}}

Tuning hyperparameters
----------------------

The default hyperparameters above will work well in most cases, but they
may not be optimal for your specific use case. There is good number of
parameters to tune, both for the
:ref:`model <architecture-{{architecture}}_model_hypers>` and the
:ref:`trainer <architecture-{{architecture}}_trainer_hypers>`. Here, we provide a
**list of the parameters that are in general the most important** (in decreasing order
of importance) for the PhACE architecture:

.. container:: mtt-hypers-remove-classname

  .. autoattribute:: {{model_hypers_path}}.radial_basis
      :no-index:

  .. autoattribute:: {{model_hypers_path}}.num_element_channels
      :no-index:

  .. autoattribute:: {{trainer_hypers_path}}.num_epochs
      :no-index:

  .. autoattribute:: {{trainer_hypers_path}}.batch_size
      :no-index:

  .. autoattribute:: {{model_hypers_path}}.num_message_passing_layers
      :no-index:

  .. autoattribute:: {{trainer_hypers_path}}.learning_rate
      :no-index:

  .. autoattribute:: {{model_hypers_path}}.cutoff
      :no-index:

  .. autoattribute:: {{model_hypers_path}}.force_rectangular
      :no-index:

  .. autoattribute:: {{model_hypers_path}}.spherical_linear_layers
      :no-index:
"""

from typing import Literal, Optional

from typing_extensions import TypedDict

from metatrain.utils.additive import FixedCompositionWeights
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification
from metatrain.utils.scaler import FixedScalerWeights


class RadialBasisHypers(TypedDict):
    """In some systems and datasets, enabling long-range Coulomb interactions
    might be beneficial for the accuracy of the model and/or
    its physical correctness."""

    max_eigenvalue: float = 25.0
    """Maximum eigenvalue for the radial basis."""

    scale: float = 0.7
    """Scaling factor for the radial basis."""

    optimizable_lengthscales: bool = False
    """Whether the length scales in the radial basis are optimizable."""


###########################
#  MODEL HYPERPARAMETERS  #
###########################


class ModelHypers(TypedDict):
    """Hyperparameters for the experimental.phace model."""

    max_correlation_order_per_layer: int = 3
    """Maximum correlation order per layer."""

    num_message_passing_layers: int = 2
    """Number of message passing layers.

    Increasing this value might increase the accuracy of the model (especially on
    larger datasets), at the expense of computational efficiency.
    """

    cutoff: float = 5.0
    """Cutoff radius for neighbor search.

    This should be set to a value after which most of the interactions
    between atoms is expected to be negligible. A lower cutoff will lead
    to faster models.
    """

    cutoff_width: float = 1.0
    """Width of the cutoff smoothing function."""

    num_element_channels: int = 128
    """Number of channels per element.

    This determines the size of the embedding used to encode the atomic species, and it
    increases or decreases the size of the internal features used in the model.
    """

    force_rectangular: bool = False
    """Makes the number of channels per irrep the same.

    This might improve accuracy with a limited increase in computational cost.
    """

    spherical_linear_layers: bool = False
    """Whether to perform linear layers in the spherical representation."""

    radial_basis: RadialBasisHypers = init_with_defaults(RadialBasisHypers)
    """Hyperparameters for the radial basis functions.

    Raising``max_eigenvalue`` from its default will increase the number of spherical
    irreducible representations (irreps) used in the model, which can improve accuracy
    at the cost of computational efficiency. Increasing this value will also increase
    the number of radial basis functions (and therefore internal features) used for each
    irrep.
    """

    nu_scaling: float = 0.1
    """Scaling for the nu term."""

    mp_scaling: float = 0.1
    """Scaling for message passing."""

    overall_scaling: float = 1.0
    """Overall scaling factor."""

    disable_nu_0: bool = True
    """Whether to disable nu=0."""

    use_sphericart: bool = False
    """Whether to use spherical Cartesian coordinates."""

    head_num_layers: int = 1
    """Number of layers in the head."""

    heads: dict[str, Literal["linear", "mlp"]] = {}
    """Heads to use in the model, with options being "linear" or "mlp"."""

    zbl: bool = False
    """Whether to use the ZBL potential in the model."""


##############################
#  TRAINER HYPERPARAMETERS   #
##############################


class TrainerHypers(TypedDict):
    """Hyperparameters for training the experimental.phace model."""

    compile: bool = True
    """Whether to use `torch.compile` during training.

    This can lead to significant speedups, but it will cause a compilation step at the
    beginning of training which might take up to 5-10 minutes, mainly depending on
    ``max_eigenvalue``.
    """

    distributed: bool = False
    """Whether to use distributed training."""

    distributed_port: int = 39591
    """Port for DDP communication."""

    batch_size: int = 8
    """Batch size for training.

    Decrease this value if you run into out-of-memory errors during training. You can
    try to increase it if your structures are very small (less than 20 atoms) and you
    have a good GPU.
    """

    num_epochs: int = 1000
    """Number of epochs to train the model.

    A larger number of epochs might lead to better accuracy. In general, if you see
    that the validation metrics are not much worse than the training ones at the end of
    training, it might be a good idea to increase this value.
    """

    learning_rate: float = 0.01
    """Learning rate for the optimizer.

    You can try to increase this value (e.g., to 0.02 or 0.03) if training is very
    slow or decrease it (e.g., to 0.005 or less) if you see that training explodes in
    the first few epochs.
    """

    warmup_fraction: float = 0.01
    """Fraction of training steps for learning rate warmup."""

    gradient_clipping: Optional[float] = None
    """Gradient clipping value. If None, no clipping is applied."""

    log_interval: int = 1
    """Interval to log metrics during training."""

    checkpoint_interval: int = 25
    """Interval to save model checkpoints."""

    scale_targets: bool = True
    """Whether to scale targets during training."""

    fixed_composition_weights: FixedCompositionWeights = {}
    """Fixed weights for atomic composition during training."""

    fixed_scaling_weights: FixedScalerWeights = {}
    """Fixed scaling weights for the model."""

    num_workers: Optional[int] = None
    """Number of workers for data loading."""

    per_structure_targets: list[str] = []
    """List of targets to calculate per-structure losses."""

    log_separate_blocks: bool = False
    """Whether to log per-block error during training."""

    log_mae: bool = False
    """Whether to log MAE alongside RMSE during training."""

    best_model_metric: Literal["rmse_prod", "mae_prod", "loss"] = "rmse_prod"
    """Metric used to select the best model checkpoint."""

    loss: str | dict[str, LossSpecification] = "mse"
    """Loss function used for training."""
