import shutil
import subprocess
from pathlib import Path

import ase.io
import numpy as np
import torch
from metatomic.torch import ModelOutput
from metatomic.torch.ase_calculator import MetatomicCalculator

from metatrain.experimental.classifier import Classifier
from metatrain.utils.data import DatasetInfo, TargetInfo


HERE = Path(__file__).parent

torch.manual_seed(42)


def test_classifier_initialization():
    """Test that the Classifier model can be initialized."""
    hypers = {
        "hidden_sizes": [64, 32],
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
    """Test that the last hidden layer acts as bottleneck."""
    hypers = {
        "hidden_sizes": [64, 32, 16],
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
    assert model.hypers["hidden_sizes"][-1] == 16


def test_classifier(monkeypatch, tmp_path):
    """
    Tests the functionalities of the Classifier model from the CLI.
    """

    monkeypatch.chdir(tmp_path)
    shutil.copy(HERE / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")
    shutil.copy(HERE / "options-pet.yaml", "options-pet.yaml")
    shutil.copy(HERE / "options-classifier.yaml", "options-classifier.yaml")

    # Add class labels to the structures (3 classes: 0, 1, 2)
    # Using one-hot encoding
    structures = ase.io.read("qm9_reduced_100.xyz", ":")
    for i, structure in enumerate(structures):
        # Assign classes in a repeating pattern
        class_id = i % 3
        # One-hot encoding for 3 classes
        if class_id == 0:
            structure.info["class_label"] = [1.0, 0.0, 0.0]
        elif class_id == 1:
            structure.info["class_label"] = [0.0, 1.0, 0.0]
        else:
            structure.info["class_label"] = [0.0, 0.0, 1.0]
    ase.io.write("qm9_reduced_100.xyz", structures)

    # 1. Train a PET model on a subset of the QM9 dataset:
    command = ["mtt", "train", "options-pet.yaml"]
    subprocess.check_call(command)

    # 2. Create the Classifier model, using `mtt train`:
    command = ["mtt", "train", "options-classifier.yaml", "-o", "model-classifier.pt"]
    subprocess.check_call(command)

    # 3. Check that the exported Classifier model from this training run works as
    #    intended, from an ASE calculator:
    check_exported_model_predictions("model-classifier.pt")

    # 4. Check that a model exported from the checkpoint also works as intended
    command = ["mtt", "export", "model-classifier.ckpt"]
    subprocess.check_call(command)
    check_exported_model_predictions("model-classifier.pt")


def check_exported_model_predictions(exported_model_path: str):
    """Check that the exported model can make predictions."""
    calc = MetatomicCalculator(exported_model_path)
    structures = ase.io.read("qm9_reduced_100.xyz", ":")[:5]

    outputs = {
        "class": ModelOutput(per_atom=False),
    }
    predictions = calc.run_model(structures, outputs)
    class_probs = predictions["class"].block().values

    # Check shape (5 structures, 3 classes)
    assert class_probs.shape == (5, 3)

    # Check that probabilities sum to 1
    assert torch.allclose(
        class_probs.sum(dim=1), torch.ones(5), atol=1e-5, rtol=1e-5
    )

    # Check that all probabilities are between 0 and 1
    assert torch.all(class_probs >= 0)
    assert torch.all(class_probs <= 1)


def test_torchscript():
    """Test that the Classifier model can be exported to TorchScript."""
    hypers = {
        "hidden_sizes": [32, 16, 2],
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

    # Try to script the model
    try:
        scripted = torch.jit.script(model)
        assert scripted is not None
    except Exception as e:
        raise AssertionError(f"Failed to script Classifier model: {e}")
