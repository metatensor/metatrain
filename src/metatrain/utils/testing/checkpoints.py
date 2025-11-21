import copy
import glob
import gzip
import logging
import os
from typing import Any, Dict

import pytest
import torch
from omegaconf import OmegaConf

from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification

from .base import ArchitectureTests


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
    incompatible_trainer_checkpoints: list[str] = []
    """A list of checkpoint paths that are known
    to be incompatible with the current trainer version when restarting."""

    @pytest.fixture
    def model_trainer(
        self, dataset_targets, DATASET_PATH, minimal_model_hypers, default_hypers
    ) -> Any:
        dataset, targets_info, dataset_info = self.get_dataset(
            dataset_targets, DATASET_PATH
        )

        model = self.model_cls(minimal_model_hypers, dataset_info)

        hypers = copy.deepcopy(default_hypers)
        hypers["training"]["num_epochs"] = 1
        loss_hypers = OmegaConf.create(
            {k: init_with_defaults(LossSpecification) for k in dataset_targets}
        )
        loss_hypers = OmegaConf.to_container(loss_hypers, resolve=True)
        hypers["training"]["loss"] = loss_hypers

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

    @pytest.mark.parametrize("context", ["restart", "finetune", "export"])
    def test_loading_old_checkpoints(
        self, default_hypers, model_trainer: Any, context: str
    ) -> None:
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
        self, monkeypatch: Any, tmp_path: str, model_trainer: Any
    ) -> None:
        """
        Test that the checkpoint did not change.

        :param monkeypatch: The pytest monkeypatch fixture.
        :param tmp_path: The pytest tmp_path fixture.
        :param model_trainer: A tuple of (model, trainer) to be tested.
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

    @pytest.mark.parametrize("context", ["finetune", "restart", "export"])
    def test_get_checkpoint(self, context, caplog, model_trainer):
        """
        Test that the checkpoint created by the model.get_checkpoint()
        function can be loaded back in all possible contexts.
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
    def test_failed_checkpoint_upgrade(self, cls_type):
        """Test error raised when trying to upgrade an invalid checkpoint version."""
        invalid_version = 99999999999999
        checkpoint = {f"{cls_type}_ckpt_version": invalid_version}

        if cls_type == "model":
            cls = self.model_cls
            version = self.model_cls.__checkpoint_version__
        else:
            cls = self.trainer_cls
            version = self.trainer_cls.__checkpoint_version__

        match = (
            f"Unable to upgrade the checkpoint: the checkpoint is using {cls_type} "
            f"version {invalid_version}, while the current {cls_type} version is "
            f"{version}."
        )
        with pytest.raises(RuntimeError, match=match):
            cls.upgrade_checkpoint(checkpoint)
