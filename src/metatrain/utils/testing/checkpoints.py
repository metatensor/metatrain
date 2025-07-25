import glob
import gzip
import os
from pathlib import Path

import pytest
import torch


def check_same_checkpoint_structure(checkpoint, reference, prefix=""):
    assert isinstance(checkpoint, dict)
    assert isinstance(reference, dict)

    for key in reference:
        if key not in checkpoint:
            raise KeyError(f"missing key from checkpoint: {prefix}.{key}")

    for key in checkpoint:
        if key not in reference:
            raise KeyError(f"new key in checkpoint: {prefix}.{key}")

    for key in reference:
        if isinstance(reference[key], dict):
            check_same_checkpoint_structure(
                checkpoint[key], reference[key], prefix=prefix + "." + str(key)
            )


def checkpoint_did_not_change(monkeypatch, tmp_path, model_trainer):
    model, trainer = model_trainer

    cwd = os.getcwd()
    monkeypatch.chdir(tmp_path)
    trainer.save_checkpoint(model, "checkpoint.ckpt")
    checkpoint = torch.load("checkpoint.ckpt", weights_only=False)
    monkeypatch.chdir(cwd)

    model_version = model.__checkpoint_version__
    trainer_version = trainer.__checkpoint_version__

    filename = Path(
        f"checkpoints/model-v{model_version}-trainer-v{trainer_version}.ckpt.gz"
    )
    if not filename.exists():
        with gzip.open(filename, "wb") as output:
            with open(os.path.join(tmp_path, "checkpoint.ckpt"), "rb") as input:
                output.write(input.read())

        raise ValueError(
            f"missing reference checkpoint for {filename.name}, we created one for you"
            "with the current state of the code. "
            f"Please move it to `{filename}` if you have no other changes to do"
        )

    else:
        with gzip.open(filename, "rb") as fd:
            reference = torch.load(fd, weights_only=False)

        try:
            check_same_checkpoint_structure(checkpoint, reference)
        except KeyError as e:
            raise ValueError(
                "checkpoint structure changed. Please increase the checkpoint "
                "version and implement checkpoint update"
            ) from e


def make_checkpoint_load_tests(DEFAULT_HYPERS):
    @pytest.mark.parametrize("context", ["restart", "finetune", "export"])
    def test_loading_old_checkpoints(model_trainer, context):
        model, trainer = model_trainer

        for path in glob.glob("checkpoints/*.ckpt.gz"):
            with gzip.open(path, "rb") as fd:
                checkpoint = torch.load(fd, weights_only=False)

            if checkpoint["model_ckpt_version"] != model.__checkpoint_version__:
                checkpoint = model.__class__.upgrade_checkpoint(checkpoint)

            model.load_checkpoint(checkpoint, context)

            if context != "export":
                if checkpoint["trainer_ckpt_version"] != trainer.__checkpoint_version__:
                    checkpoint = trainer.__class__.upgrade_checkpoint(checkpoint)

                trainer.load_checkpoint(checkpoint, DEFAULT_HYPERS, context)

    return test_loading_old_checkpoints
