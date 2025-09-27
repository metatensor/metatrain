import glob
import gzip
import os

import pytest
import torch


ALLOWED_NEW_KEYS_CONDITIONS = [
    # torch added this key in LambdaLR
    lambda prefix, key: "scheduler" in f"{prefix}.{key}" and key == "_is_initial"
]


def check_same_checkpoint_structure(checkpoint, reference, prefix=""):
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


def checkpoint_did_not_change(monkeypatch, tmp_path, model_trainer):
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
            print(e)
            raise ValueError(
                "checkpoint structure changed. Please increase the checkpoint "
                "version and implement checkpoint update"
            ) from e


def make_checkpoint_load_tests(
    DEFAULT_HYPERS, *, incompatible_trainer_checkpoints=None
):
    if incompatible_trainer_checkpoints is None:
        incompatible_trainer_checkpoints = []

    @pytest.mark.parametrize("context", ["restart", "finetune", "export"])
    def test_loading_old_checkpoints(model_trainer, context):
        model, trainer = model_trainer

        for path in glob.glob("checkpoints/*.ckpt.gz"):
            if path in incompatible_trainer_checkpoints and context == "restart":
                continue

            with gzip.open(path, "rb") as fd:
                checkpoint = torch.load(fd, weights_only=False)

            if checkpoint["model_ckpt_version"] != model.__checkpoint_version__:
                checkpoint = model.__class__.upgrade_checkpoint(checkpoint)

            model.load_checkpoint(checkpoint, context)

            if context == "restart":
                if checkpoint["trainer_ckpt_version"] != trainer.__checkpoint_version__:
                    checkpoint = trainer.__class__.upgrade_checkpoint(checkpoint)

                trainer.load_checkpoint(checkpoint, DEFAULT_HYPERS, context)

    return test_loading_old_checkpoints
