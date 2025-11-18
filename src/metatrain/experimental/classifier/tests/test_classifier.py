import pytest
import torch

from metatrain.experimental.classifier import Classifier
from metatrain.utils.data import DatasetInfo, TargetInfo


def test_classifier_initialization():
    """Test that the Classifier model can be initialized."""
    hypers = {
        "hidden_sizes": [64, 32],
        "bottleneck_size": None,
    }

    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 6, 8],
        targets={
            "class": TargetInfo(
                quantity="",
                unit="",
            )
        },
    )

    model = Classifier(hypers, dataset_info)
    assert model is not None
    assert model.hypers == hypers
    assert model.dataset_info == dataset_info


def test_classifier_with_bottleneck():
    """Test that the Classifier model can be initialized with a bottleneck."""
    hypers = {
        "hidden_sizes": [64, 32],
        "bottleneck_size": 16,
    }

    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 6, 8],
        targets={
            "class": TargetInfo(
                quantity="",
                unit="",
            )
        },
    )

    model = Classifier(hypers, dataset_info)
    assert model is not None
    assert model.hypers["bottleneck_size"] == 16
