"""
Classifier
==========

The Classifier architecture is an experimental "wrapper" architecture that enables
classification tasks on top of pre-trained atomistic models. It takes a pre-trained
checkpoint, freezes its backbone, and trains a small multi-layer perceptron (MLP) on
top of the features extracted from the backbone.

The model extracts per-atom features from the frozen backbone, averages them to get
system-level representations, and then passes them through the MLP for classification.
The targets should be class probabilities as vectors, supporting both one-hot encodings
(e.g., [1.0, 0.0, 0.0]) and soft/fractional targets (e.g., [0.7, 0.2, 0.1]). The loss
function is a standard cross-entropy loss for classification.

The last layer in `hidden_sizes` can be set to a small value if the goal is to use it to
extract features for low-dimensional visualization and/or collective variables.

{{SECTION_INSTALLATION}}

{{SECTION_DEFAULT_HYPERS}}

{{SECTION_MODEL_HYPERS}}

"""

from typing import Optional

from typing_extensions import TypedDict


class ModelHypers(TypedDict):
    """Hyperparameters for the Classifier model."""

    hidden_sizes: list[int] = [64, 64]
    """List of hidden layer sizes for the MLP.
    For example, [64, 32] creates a 2-layer MLP with 64 and 32 neurons
    respectively. The last hidden size should be set to a small number
    (generally one or two) if the goal is to extract collective variables.
    """


class TrainerHypers(TypedDict):
    """Hyperparameters for the Classifier trainer."""

    batch_size: int = 32
    """Batch size for training."""
    num_epochs: int = 100
    """Number of training epochs."""
    learning_rate: float = 0.001
    """Learning rate for the optimizer."""
    warmup_fraction: float = 0.1
    """Fraction of total training steps used for learning rate warmup.
    The learning rate increases linearly from 0 to the base learning rate
    during this period, then follows a cosine annealing schedule.
    """
    model_checkpoint: Optional[str] = None
    """Path to the pre-trained model checkpoint.
    This checkpoint's backbone will be frozen and used for feature extraction.
    """
    log_interval: int = 1
    """Interval for logging training progress (in epochs)."""
    checkpoint_interval: int = 100
    """Interval for saving checkpoints during training (in epochs)."""
