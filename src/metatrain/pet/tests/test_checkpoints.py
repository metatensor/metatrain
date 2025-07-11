import copy
import glob
import gzip

import pytest
import torch

from metatrain.pet import PET, Trainer
from metatrain.utils.data import DatasetInfo, get_atomic_types, get_dataset

from . import DATASET_PATH, DEFAULT_HYPERS, MODEL_HYPERS


@pytest.fixture(scope="module")
def model_trainer():
    energy_target = {
        "quantity": "energy",
        "read_from": DATASET_PATH,
        "reader": "ase",
        "key": "U0",
        "unit": "eV",
        "type": "scalar",
        "per_atom": False,
        "num_subtargets": 1,
        "forces": False,
        "stress": False,
        "virial": False,
    }

    dataset, targets_info, _ = get_dataset(
        {
            "systems": {
                "read_from": DATASET_PATH,
                "reader": "ase",
            },
            "targets": {
                "energy": energy_target,
            },
        }
    )

    dataset_info = DatasetInfo(
        length_unit="",
        atomic_types=get_atomic_types(dataset),
        targets=targets_info,
    )

    # minimize the size of the checkpoint on disk
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["d_pet"] = 1
    hypers["d_head"] = 1
    hypers["d_feedforward"] = 1
    hypers["num_heads"] = 1
    hypers["num_attention_layers"] = 1
    hypers["num_gnn_layers"] = 1

    model = PET(hypers, dataset_info)

    hypers = copy.deepcopy(DEFAULT_HYPERS)
    hypers["training"]["num_epochs"] = 1
    trainer = Trainer(hypers["training"])

    trainer.train(
        model,
        dtype=model.__supported_dtypes__[0],
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir="",
    )

    return model, trainer


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


def test_checkpoint_did_not_change(monkeypatch, tmp_path, model_trainer):
    model, trainer = model_trainer
    version = model.__checkpoint_version__
    with gzip.open(f"checkpoints/v{version}.ckpt.gz", "rb") as fd:
        reference = torch.load(fd, weights_only=False)

    monkeypatch.chdir(tmp_path)
    trainer.save_checkpoint(model, "checkpoint.ckpt")

    checkpoint = torch.load("checkpoint.ckpt", weights_only=False)

    try:
        check_same_checkpoint_structure(checkpoint, reference)
    except KeyError as e:
        raise ValueError(
            "checkpoint structure changed. Please increase the checkpoint "
            "version and implement checkpoint update"
        ) from e


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
