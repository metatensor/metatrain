import copy
import shutil
import subprocess
import tempfile
from pathlib import Path

import ase.io
import pytest
import torch
from metatomic.torch import ModelOutput
from metatomic.torch.ase_calculator import MetatomicCalculator
from omegaconf import OmegaConf

from metatrain.experimental.classifier import Classifier
from metatrain.pet import PET
from metatrain.pet import Trainer as PETTrainer
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification
from metatrain.utils.testing import ArchitectureTests, CheckpointTests


HERE = Path(__file__).parent

torch.manual_seed(42)


class ClassifierTests(ArchitectureTests):
    architecture = "experimental.classifier"


class TestCheckpoints(CheckpointTests, ClassifierTests):
    @pytest.fixture
    def model_trainer(
        self, dataset_targets, dataset_path, model_hypers, default_hypers
    ):
        dataset, targets_info, dataset_info = self.get_dataset(
            dataset_targets, dataset_path
        )

        hypers = copy.deepcopy(get_default_hypers("pet"))

        pet_model_hypers = hypers["model"]
        pet_model_hypers["d_pet"] = 1
        pet_model_hypers["d_head"] = 1
        pet_model_hypers["d_feedforward"] = 1
        pet_model_hypers["num_heads"] = 1
        pet_model_hypers["num_attention_layers"] = 1
        pet_model_hypers["num_gnn_layers"] = 1

        pet_model = PET(pet_model_hypers, dataset_info)

        hypers = copy.deepcopy(hypers)
        hypers["training"]["num_epochs"] = 1
        loss_hypers = OmegaConf.create(
            {k: init_with_defaults(LossSpecification) for k in dataset_targets}
        )
        loss_hypers = OmegaConf.to_container(loss_hypers, resolve=True)
        hypers["training"]["loss"] = loss_hypers

        trainer = PETTrainer(hypers["training"])

        trainer.train(
            pet_model,
            dtype=pet_model.__supported_dtypes__[0],
            devices=[torch.device("cpu")],
            train_datasets=[dataset],
            val_datasets=[dataset],
            checkpoint_dir="",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_checkpoint(pet_model, f"{tmpdir}/pet_checkpoint.ckpt")

            # train Classifier model
            hypers = copy.deepcopy(model_hypers)

            model = self.model_cls(hypers, dataset_info)

            hypers = copy.deepcopy(default_hypers)
            hypers["training"]["model_checkpoint"] = f"{tmpdir}/pet_checkpoint.ckpt"

            trainer = self.trainer_cls(hypers["training"])
            trainer.train(
                model,
                dtype=model.__supported_dtypes__[0],
                devices=[torch.device("cpu")],
                train_datasets=[dataset],
                val_datasets=[dataset],
                checkpoint_dir="",
            )

        return model, trainer


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


def test_checkpoint_export_from_ckpt(monkeypatch, tmp_path):
    """
    Test that a Classifier model can be exported from a checkpoint file.
    Similar to the LLPR tests, this checks that `mtt export` works on the .ckpt file.
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

    # Train the Classifier model (outputs model-classifier.ckpt checkpoint)
    subprocess.check_call(
        ["mtt", "train", "options-classifier.yaml", "-o", "model-classifier.pt"]
    )

    # Check that a model exported from the checkpoint also works as intended
    subprocess.check_call(["mtt", "export", "model-classifier.ckpt"])

    # Verify the exported model works with MetatomicCalculator
    calc = MetatomicCalculator("model-classifier.pt")
    test_structures = ase.io.read("qm9_reduced_100.xyz", ":")[:3]

    outputs = {"mtt::class": ModelOutput()}
    predictions = calc.run_model(test_structures, outputs)
    class_probs = predictions["mtt::class"].block().values

    # Check shape (3 structures, 3 classes)
    assert class_probs.shape == (3, 3)

    # Check that probabilities sum to 1
    assert torch.allclose(
        class_probs.sum(dim=1),
        torch.ones(3, device=class_probs.device),
        atol=1e-5,
        rtol=1e-5,
    )
