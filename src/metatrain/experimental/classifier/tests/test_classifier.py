import shutil
import subprocess
from pathlib import Path

import ase.io
import pytest
import torch
from metatomic.torch import ModelOutput
from metatomic.torch.ase_calculator import MetatomicCalculator

from metatrain.experimental.classifier import Classifier
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_generic_target_info


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
            "mtt::class": get_generic_target_info(
                "mtt::class",
                {
                    "quantity": "",
                    "unit": "",
                    "num_subtargets": 3,
                    "type": "scalar",
                    "per_atom": False,
                },
            )
        },
    )

    model = Classifier(hypers, dataset_info)
    assert model is not None
    assert model.hypers == hypers
    assert model.dataset_info == dataset_info


def test_classifier(monkeypatch, tmp_path):
    """
    Tests the functionalities of the Classifier model from the CLI.
    Note that this automatically tests torchscript export and ASE calculator.
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
    calc = MetatomicCalculator("model-classifier.pt")
    structures = ase.io.read("qm9_reduced_100.xyz", ":")[:5]

    outputs = {
        "mtt::class": ModelOutput(),
    }
    predictions = calc.run_model(structures, outputs)
    class_probs = predictions["mtt::class"].block().values

    # Check shape (5 structures, 3 classes)
    assert class_probs.shape == (5, 3)

    # Check that probabilities sum to 1
    assert torch.allclose(
        class_probs.sum(dim=1),
        torch.ones(5, device=class_probs.device),
        atol=1e-5,
        rtol=1e-5,
    )

    # Check that all probabilities are between 0 and 1
    assert torch.all(class_probs >= 0)
    assert torch.all(class_probs <= 1)


def test_checkpoint_export(monkeypatch, tmp_path):
    """
    Test that the Classifier model checkpoint can be saved and loaded for export.
    """
    monkeypatch.chdir(tmp_path)
    shutil.copy(HERE / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")
    shutil.copy(HERE / "options-pet.yaml", "options-pet.yaml")
    shutil.copy(HERE / "options-classifier.yaml", "options-classifier.yaml")

    # Add class labels to the structures
    structures = ase.io.read("qm9_reduced_100.xyz", ":")
    for i, structure in enumerate(structures):
        class_id = i % 3
        if class_id == 0:
            structure.info["class_label"] = [1.0, 0.0, 0.0]
        elif class_id == 1:
            structure.info["class_label"] = [0.0, 1.0, 0.0]
        else:
            structure.info["class_label"] = [0.0, 0.0, 1.0]
    ase.io.write("qm9_reduced_100.xyz", structures)

    # Train a PET model first
    subprocess.check_call(["mtt", "train", "options-pet.yaml"])

    # Train the Classifier model
    subprocess.check_call(
        ["mtt", "train", "options-classifier.yaml", "-o", "classifier.pt"]
    )

    # Load the checkpoint and verify it works
    checkpoint = torch.load("classifier.pt", weights_only=False)
    assert "model_ckpt_version" in checkpoint
    assert checkpoint["model_ckpt_version"] == Classifier.__checkpoint_version__

    # Test that the model can be loaded from checkpoint in export context
    model = Classifier.load_checkpoint(checkpoint, "export")
    assert model is not None

    # Export the model and verify it works
    exported = model.export()
    assert exported is not None


def test_checkpoint_finetune_error(monkeypatch, tmp_path):
    """
    Test that attempting to finetune from a Classifier checkpoint raises an error.
    """
    monkeypatch.chdir(tmp_path)
    shutil.copy(HERE / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")
    shutil.copy(HERE / "options-pet.yaml", "options-pet.yaml")
    shutil.copy(HERE / "options-classifier.yaml", "options-classifier.yaml")

    # Add class labels to the structures
    structures = ase.io.read("qm9_reduced_100.xyz", ":")
    for i, structure in enumerate(structures):
        class_id = i % 3
        if class_id == 0:
            structure.info["class_label"] = [1.0, 0.0, 0.0]
        elif class_id == 1:
            structure.info["class_label"] = [0.0, 1.0, 0.0]
        else:
            structure.info["class_label"] = [0.0, 0.0, 1.0]
    ase.io.write("qm9_reduced_100.xyz", structures)

    # Train a PET model first
    subprocess.check_call(["mtt", "train", "options-pet.yaml"])

    # Train the Classifier model
    subprocess.check_call(
        ["mtt", "train", "options-classifier.yaml", "-o", "classifier.pt"]
    )

    # Load the checkpoint
    checkpoint = torch.load("classifier.pt", weights_only=False)

    # Test that finetune raises an error
    with pytest.raises(NotImplementedError, match="Finetuning.*not supported"):
        Classifier.load_checkpoint(checkpoint, "finetune")


def test_checkpoint_restart_error(monkeypatch, tmp_path):
    """
    Test that attempting to restart from a Classifier checkpoint raises an error.
    """
    monkeypatch.chdir(tmp_path)
    shutil.copy(HERE / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")
    shutil.copy(HERE / "options-pet.yaml", "options-pet.yaml")
    shutil.copy(HERE / "options-classifier.yaml", "options-classifier.yaml")

    # Add class labels to the structures
    structures = ase.io.read("qm9_reduced_100.xyz", ":")
    for i, structure in enumerate(structures):
        class_id = i % 3
        if class_id == 0:
            structure.info["class_label"] = [1.0, 0.0, 0.0]
        elif class_id == 1:
            structure.info["class_label"] = [0.0, 1.0, 0.0]
        else:
            structure.info["class_label"] = [0.0, 0.0, 1.0]
    ase.io.write("qm9_reduced_100.xyz", structures)

    # Train a PET model first
    subprocess.check_call(["mtt", "train", "options-pet.yaml"])

    # Train the Classifier model
    subprocess.check_call(
        ["mtt", "train", "options-classifier.yaml", "-o", "classifier.pt"]
    )

    # Load the checkpoint
    checkpoint = torch.load("classifier.pt", weights_only=False)

    # Test that restart raises an error
    with pytest.raises(NotImplementedError, match="Restarting.*not supported"):
        Classifier.load_checkpoint(checkpoint, "restart")


def test_failed_model_checkpoint_upgrade():
    """Test that upgrading an invalid model checkpoint version raises an error."""
    invalid_checkpoint = {"model_ckpt_version": 99999999}

    with pytest.raises(RuntimeError, match="Unable to upgrade the checkpoint"):
        Classifier.upgrade_checkpoint(invalid_checkpoint)
