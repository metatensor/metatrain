"""
Classifier
==========

The Classifier architecture is an experimental "wrapper" architecture that enables
classification tasks on top of pre-trained atomistic models. It takes a pre-trained
checkpoint, freezes its backbone, and trains a small multi-layer perceptron (MLP) on
top of the features extracted from the backbone. The targets should be class labels
specified as single floats that are actually integers (0.0, 1.0, 2.0, etc.), so that
we can re-use the metatrain infrastructure. The loss function is a negative
log-likelihood (NLL) classification loss.

{{SECTION_INSTALLATION}}

{{SECTION_DEFAULT_HYPERS}}

{{SECTION_MODEL_HYPERS}}

"""

from typing import Optional

from typing_extensions import TypedDict


class ModelHypers(TypedDict):
    """Hyperparameters for the Classifier model."""

    hidden_sizes: list[int]
    """List of hidden layer sizes for the MLP. For example, [64, 32] creates
    a 2-layer MLP with 64 and 32 neurons respectively."""

    bottleneck_size: Optional[int]
    """Optional bottleneck layer size before the final classification layer.
    This layer can be used to extract features/collective variables.
    If None, no bottleneck is used."""


class TrainerHypers(TypedDict):
    """Hyperparameters for the Classifier trainer."""

    batch_size: int = 32
    """Batch size for training."""

    num_epochs: int = 100
    """Number of training epochs."""

    learning_rate: float = 0.001
    """Learning rate for the optimizer."""

    model_checkpoint: Optional[str] = None
    """Path to the pre-trained model checkpoint. This checkpoint's backbone
    will be frozen and used for feature extraction."""

    weight_decay: float = 0.0
    """Weight decay (L2 regularization) for the optimizer."""

    log_interval: int = 10
    """Interval for logging training progress (in epochs)."""

    checkpoint_interval: int = 10
    """Interval for saving checkpoints during training (in epochs)."""
