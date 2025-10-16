import copy
import functools as ft
import logging
import re
import tempfile

import pytest
import torch
from omegaconf import OmegaConf

from metatrain.llpr import LLPRUncertaintyModel
from metatrain.llpr import Trainer as LLPRTrainer
from metatrain.pet import PET
from metatrain.pet import Trainer as PETTrainer
from metatrain.utils.data import DatasetInfo, get_atomic_types, get_dataset
from metatrain.utils.omegaconf import CONF_LOSS
from metatrain.utils.testing.checkpoints import (
    checkpoint_did_not_change,
    make_checkpoint_load_tests,
)

from . import (
    DATASET_PATH,
    DEFAULT_HYPERS_LLPR,
    DEFAULT_HYPERS_PET,
    MODEL_HYPERS_LLPR,
    MODEL_HYPERS_PET,
)


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
    hypers = copy.deepcopy(MODEL_HYPERS_PET)
    hypers["d_pet"] = 1
    hypers["d_head"] = 1
    hypers["d_feedforward"] = 1
    hypers["num_heads"] = 1
    hypers["num_attention_layers"] = 1
    hypers["num_gnn_layers"] = 1

    model = PET(hypers, dataset_info)

    hypers = copy.deepcopy(DEFAULT_HYPERS_PET)
    hypers["training"]["num_epochs"] = 1
    loss_hypers = OmegaConf.create({"energy": CONF_LOSS.copy()})
    loss_hypers = OmegaConf.to_container(loss_hypers, resolve=True)
    hypers["training"]["loss"] = loss_hypers

    trainer = PETTrainer(hypers["training"])

    trainer.train(
        model,
        dtype=model.__supported_dtypes__[0],
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir="",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.save_checkpoint(model, f"{tmpdir}/pet_checkpoint.ckpt")

        # train LLPR model
        hypers = copy.deepcopy(MODEL_HYPERS_LLPR)
        hypers["ensembles"]["means"] = {
            "energy": [
                "node_last_layers.energy.0.energy___0.weight",
                "edge_last_layers.energy.0.energy___0.weight",
            ]
        }
        hypers["ensembles"]["num_members"] = {"energy": 8}

        model = LLPRUncertaintyModel(hypers, dataset_info)

        hypers = copy.deepcopy(DEFAULT_HYPERS_LLPR)
        hypers["training"]["model_checkpoint"] = f"{tmpdir}/pet_checkpoint.ckpt"

        trainer = LLPRTrainer(hypers["training"])
        trainer.train(
            model,
            dtype=model.__supported_dtypes__[0],
            devices=[torch.device("cpu")],
            train_datasets=[dataset],
            val_datasets=[dataset],
            checkpoint_dir="",
        )

    return model, trainer


test_checkpoint_did_not_change = checkpoint_did_not_change


def ignore_exception(exc_type, match: str):
    def decorator(func):
        @ft.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exc_type as e:
                if re.search(match, str(e)):
                    return  # count as PASS
                raise

        return wrapper

    return decorator


test_loading_old_checkpoints = ignore_exception(
    NotImplementedError, "Restarting from the LLPR checkpoint is not supported."
)(make_checkpoint_load_tests(DEFAULT_HYPERS_LLPR))


@pytest.mark.parametrize("context", ["finetune", "restart", "export"])
def test_get_checkpoint(model_trainer, context, caplog):
    """
    Test that the checkpoint created by the model.get_checkpoint()
    function can be loaded back in all possible contexts.
    """
    model, _ = model_trainer

    checkpoint = model.get_checkpoint()

    caplog.set_level(logging.INFO)

    if context == "restart":
        with pytest.raises(
            NotImplementedError,
            match="Restarting from the LLPR checkpoint is not supported.",
        ):
            LLPRUncertaintyModel.load_checkpoint(checkpoint, context)
        return
    else:
        LLPRUncertaintyModel.load_checkpoint(checkpoint, context)

    if context == "restart":
        assert "Using latest model from epoch None" in caplog.text
    else:
        assert "Using best model from epoch None" in caplog.text


@pytest.mark.parametrize("cls_type", ["model", "trainer"])
def test_failed_checkpoint_upgrade(cls_type):
    """Test error raised when trying to upgrade an invalid checkpoint version."""
    checkpoint = {f"{cls_type}_ckpt_version": 9999}

    if cls_type == "model":
        cls = LLPRUncertaintyModel
        version = LLPRUncertaintyModel.__checkpoint_version__
    else:
        cls = LLPRTrainer
        version = LLPRTrainer.__checkpoint_version__

    match = (
        f"Unable to upgrade the checkpoint: the checkpoint is using {cls_type} version "
        f"9999, while the current {cls_type} version is {version}."
    )
    with pytest.raises(RuntimeError, match=match):
        cls.upgrade_checkpoint(checkpoint)
