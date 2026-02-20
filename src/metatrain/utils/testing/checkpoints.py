import copy
import glob
import gzip
import logging
import os
from typing import Any, Dict, Literal

import pytest
import torch
from omegaconf import OmegaConf

from metatrain.utils.abc import ModelInterface, TrainerInterface
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification

from .architectures import ArchitectureTests


ALLOWED_NEW_KEYS_CONDITIONS = [
    # torch added this key in LambdaLR
    lambda prefix, key: "scheduler" in f"{prefix}.{key}" and key == "_is_initial"
]


def check_same_checkpoint_structure(
    checkpoint: Dict[str, Any], reference: Dict[str, Any], prefix: str = ""
) -> None:
    """
    Check that the structure of two checkpoints is the same.

    :param checkpoint: The checkpoint to be checked.
    :param reference: The reference checkpoint.
    :param prefix: The prefix to be added to the keys in the error messages.
    """
    assert isinstance(checkpoint, dict)
    assert isinstance(reference, dict)

    for key in reference:
        if key not in checkpoint:
            raise KeyError(f"missing key from checkpoint: {prefix}.{key}")

    for key in checkpoint:
        if any(cond(prefix, key) for cond in ALLOWED_NEW_KEYS_CONDITIONS):
            continue
        if key not in reference:
            raise KeyError(f"new key in checkpoint: {prefix}.{key}")

    for key in reference:
        if isinstance(reference[key], dict):
            check_same_checkpoint_structure(
                checkpoint[key], reference[key], prefix=prefix + "." + str(key)
            )


class CheckpointTests(ArchitectureTests):
    """Test suite for model and trainer checkpoints.

    This test suite verifies that the checkpoints for the architecture
    follow the expected behavior of ``metatrain`` checkpoints.
    """

    incompatible_trainer_checkpoints: list[str] = []
    """A list of checkpoint paths that are known
    to be incompatible with the current trainer version when restarting.

    This should be overriden in subclasses.
    """

    @pytest.fixture
    def model_trainer(
        self,
        dataset_path: str,
        dataset_targets: dict,
        minimal_model_hypers: dict,
        default_hypers: dict,
    ) -> tuple[ModelInterface, TrainerInterface]:
        """Fixture that returns a trained model and trainer.

        The model and trainer are used in the test suite to verify checkpoint
        functionality.

        :param dataset_path: The path to the dataset file to train on.
        :param dataset_targets: The targets that the dataset contains.
        :param minimal_model_hypers: Hyperparameters to initialize the model.
            These should give the smallest possible model to use as little
            disk space as possible when saving checkpoints.
        :param default_hypers: Default hyperparameters to initialize the trainer.

        :return: A tuple containing the trained model and the trainer.
        """
        # Load dataset
        dataset, targets_info, dataset_info = self.get_dataset(
            dataset_targets, dataset_path
        )

        # Initialize model
        model = self.model_cls(minimal_model_hypers, dataset_info)

        # Set the training hyperparameters:
        #  - Just 1 epoch to keep the test fast
        #  - Default loss for each target
        hypers = copy.deepcopy(default_hypers)
        hypers["training"]["num_epochs"] = 1
        loss_hypers = OmegaConf.create(
            {k: init_with_defaults(LossSpecification) for k in dataset_targets}
        )
        loss_hypers = OmegaConf.to_container(loss_hypers, resolve=True)
        hypers["training"]["loss"] = loss_hypers

        # Initialize trainer
        trainer = self.trainer_cls(hypers["training"])

        # Train the model.
        trainer.train(
            model,
            dtype=model.__supported_dtypes__[0],
            devices=[torch.device("cpu")],
            train_datasets=[dataset],
            val_datasets=[dataset],
            checkpoint_dir="",
        )

        return model, trainer

    @pytest.mark.parametrize("context", ["restart", "finetune", "export"])
    def test_loading_old_checkpoints(
        self,
        default_hypers: dict,
        model_trainer: tuple[ModelInterface, TrainerInterface],
        context: Literal["restart", "finetune", "export"],
    ) -> None:
        """Tests that checkpoints from previous versions can be loaded.

        This test goes through all the checkpoint files in the
        ``checkpoints/`` folder of the current directory (presumably the
        architecture's tests folder) and tries to load them in the current
        model and trainer.

        The test skips trainer checkpoints that are listed in this class's
        ``incompatible_trainer_checkpoints`` attribute when the context is
        ``restart``.

        :param default_hypers: Default hyperparameters to initialize the trainer.
        :param model_trainer: Model and trainer to be used for loading the checkpoints.
        :param context: The context in which to load the checkpoint.
        """
        model, trainer = model_trainer

        for path in glob.glob("checkpoints/*.ckpt.gz"):
            if path in self.incompatible_trainer_checkpoints and context == "restart":
                continue

            with gzip.open(path, "rb") as fd:
                checkpoint = torch.load(fd, weights_only=False)

            if checkpoint["model_ckpt_version"] != model.__checkpoint_version__:
                checkpoint = model.__class__.upgrade_checkpoint(checkpoint)
            model.load_checkpoint(checkpoint, context)

            if context == "restart":
                if checkpoint["trainer_ckpt_version"] != trainer.__checkpoint_version__:
                    checkpoint = trainer.__class__.upgrade_checkpoint(checkpoint)

                trainer.load_checkpoint(checkpoint, default_hypers, context)

    def test_checkpoint_did_not_change(
        self,
        monkeypatch: Any,
        tmp_path: str,
        model_trainer: tuple[ModelInterface, TrainerInterface],
    ) -> None:
        """
        Test that the checkpoint did not change.

        This test gets the current version of the model and trainer,
        and loads the checkpoint for that version from the ``checkpoints/``
        folder. If that checkpoint is not compatible with the current code,
        this means that the checkpoint version of either the model or the
        trainer needs to be bumped.

        :param monkeypatch: The pytest monkeypatch fixture.
        :param tmp_path: The pytest tmp_path fixture.
        :param model_trainer: Model and trainer to test.
        """
        model, trainer = model_trainer

        cwd = os.getcwd()
        monkeypatch.chdir(tmp_path)
        trainer.save_checkpoint(model, "checkpoint.ckpt")

        checkpoint = torch.load("checkpoint.ckpt", weights_only=False)
        monkeypatch.chdir(cwd)

        model_version = model.__checkpoint_version__
        trainer_version = trainer.__checkpoint_version__

        ckpt_name = f"model-v{model_version}_trainer-v{trainer_version}.ckpt.gz"
        ckpt_path = f"checkpoints/{ckpt_name}"

        if not os.path.exists(ckpt_path):
            with gzip.open(ckpt_name, "wb") as output:
                with open(os.path.join(tmp_path, "checkpoint.ckpt"), "rb") as input:
                    output.write(input.read())

            raise ValueError(
                f"missing reference checkpoint for model version {model_version} and "
                f"trainer version {trainer_version}, we created one for you with the "
                f"current state of the code. Please move it to {ckpt_path} if you "
                "have no other changes to do."
            )

        else:
            with gzip.open(ckpt_path, "rb") as fd:
                reference = torch.load(fd, weights_only=False)

            try:
                check_same_checkpoint_structure(checkpoint, reference)
            except KeyError as e:
                raise ValueError(
                    "checkpoint structure changed. Please increase the checkpoint "
                    "version and implement checkpoint update"
                ) from e

        # Ensure that the size of the checkpoint is <50kb
        if (size := os.path.getsize(ckpt_path)) > 50 * 1024:
            raise ValueError(
                f"Checkpoint size at {ckpt_path} is too large ({size} bytes), please "
                "reduce it to <50kb"
            )

    @pytest.mark.parametrize("context", ["finetune", "restart", "export"])
    def test_get_checkpoint(
        self,
        context: Literal["finetune", "restart", "export"],
        caplog: Any,
        model_trainer: tuple[ModelInterface, TrainerInterface],
    ) -> None:
        """
        Test that the checkpoint created by the ``model.get_checkpoint()``
        function can be loaded back in all possible contexts.

        This test can fail either if the model is unable to produce
        checkpoints, or if the generated checkpoint can't be loaded back
        by the model in the specified context.

        :param context: The context in which to load the generated checkpoint.
        :param caplog: The pytest caplog fixture.
        :param model_trainer: Model and trainer to be used for the test.
        """
        model, _ = model_trainer
        checkpoint = model.get_checkpoint()

        caplog.set_level(logging.INFO)
        self.model_cls.load_checkpoint(checkpoint, context)

        if context == "restart":
            assert "Using latest model from epoch None" in caplog.text
        else:
            assert "Using best model from epoch None" in caplog.text

    @pytest.mark.parametrize("cls_type", ["model", "trainer"])
    def test_failed_checkpoint_upgrade(
        self, cls_type: Literal["model", "trainer"]
    ) -> None:
        """Test error raised when trying to upgrade an invalid checkpoint version.

        This test creates a checkpoint with an invalid version number and
        tries to upgrade it using the corresponding class.

        If this test fails, it likely means that you are not raising the
        error in your model/trainer's ``upgrade_checkpoint`` method when
        the checkpoint version is not recognized. To raise the appropiate
        error:

        .. code-block:: python

            cls_type = "model"  # or "trainer"

            raise RuntimeError(
                f"Unable to upgrade the checkpoint: the checkpoint is using {cls_type} "
                f"version {checkpoint_version}, while the current {cls_type} version "
                f"is {self.__class__.__checkpoint_version__}."
            )

        :param cls_type: The class type to test.
        """
        invalid_version = 99999999999999
        checkpoint = {f"{cls_type}_ckpt_version": invalid_version}

        cls = self.model_cls if cls_type == "model" else self.trainer_cls

        version = cls.__checkpoint_version__

        match = (
            f"Unable to upgrade the checkpoint: the checkpoint is using {cls_type} "
            f"version {invalid_version}, while the current {cls_type} version is "
            f"{version}."
        )
        with pytest.raises(RuntimeError, match=match):
            cls.upgrade_checkpoint(checkpoint)
