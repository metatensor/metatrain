import copy
import logging

import pytest
import torch
from omegaconf import OmegaConf

from metatrain.experimental.flashmd import FlashMD, Trainer
from metatrain.utils.data import (
    DatasetInfo,
    get_atomic_types,
    get_dataset,
)
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.omegaconf import CONF_LOSS
from metatrain.utils.testing.checkpoints import (
    checkpoint_did_not_change,
    make_checkpoint_load_tests,
)

from . import DEFAULT_HYPERS, MODEL_HYPERS


DEFAULT_HYPERS = copy.deepcopy(DEFAULT_HYPERS)
DEFAULT_HYPERS["training"]["timestep"] = 30.0
DEFAULT_HYPERS["training"]["batch_size"] = 1


@pytest.fixture(scope="module")
def model_trainer():
    positions_target = {
        "quantity": "position",
        "read_from": "data/flashmd.xyz",
        "reader": "ase",
        "key": "future_positions",
        "unit": "A",
        "type": {
            "cartesian": {
                "rank": 1,
            }
        },
        "per_atom": True,
        "num_subtargets": 1,
    }

    momenta_target = {
        "quantity": "momentum",
        "read_from": "data/flashmd.xyz",
        "reader": "ase",
        "key": "future_momenta",
        "unit": "(eV*u)^1/2",
        "type": {
            "cartesian": {
                "rank": 1,
            }
        },
        "per_atom": True,
        "num_subtargets": 1,
    }

    dataset, targets_info, _ = get_dataset(
        {
            "systems": {
                "read_from": "data/flashmd.xyz",
                "reader": "ase",
            },
            "targets": {
                "positions": positions_target,
                "momenta": momenta_target,
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
    hypers["d_node"] = 1
    hypers["d_feedforward"] = 1
    hypers["num_heads"] = 1
    hypers["num_attention_layers"] = 1
    hypers["num_gnn_layers"] = 1

    model = FlashMD(hypers, dataset_info)

    hypers = copy.deepcopy(DEFAULT_HYPERS)
    hypers["training"]["num_epochs"] = 1
    loss_hypers = OmegaConf.create(
        {"positions": CONF_LOSS.copy(), "momenta": CONF_LOSS.copy()}
    )
    loss_hypers = OmegaConf.to_container(loss_hypers, resolve=True)
    hypers["training"]["loss"] = loss_hypers

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


test_checkpoint_did_not_change = pytest.mark.filterwarnings(
    "ignore:custom data:UserWarning"
)(checkpoint_did_not_change)

test_loading_old_checkpoints = pytest.mark.filterwarnings(
    "ignore:custom data:UserWarning"
)(make_checkpoint_load_tests(DEFAULT_HYPERS))


@pytest.mark.parametrize("context", ["finetune", "restart", "export"])
def test_get_checkpoint(context, caplog):
    """
    Test that the checkpoint created by the model.get_checkpoint()
    function can be loaded back in all possible contexts.
    """
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
    )
    model = FlashMD(MODEL_HYPERS, dataset_info)
    checkpoint = model.get_checkpoint()

    caplog.set_level(logging.INFO)
    FlashMD.load_checkpoint(checkpoint, context)

    if context == "restart":
        assert "Using latest model from epoch None" in caplog.text
    else:
        assert "Using best model from epoch None" in caplog.text


@pytest.mark.parametrize("cls_type", ["model", "trainer"])
def test_failed_checkpoint_upgrade(cls_type):
    """Test error raised when trying to upgrade an invalid checkpoint version."""
    checkpoint = {f"{cls_type}_ckpt_version": 9999}

    if cls_type == "model":
        cls = FlashMD
        version = FlashMD.__checkpoint_version__
    else:
        cls = Trainer
        version = Trainer.__checkpoint_version__

    match = (
        f"Unable to upgrade the checkpoint: the checkpoint is using {cls_type} version "
        f"9999, while the current {cls_type} version is {version}."
    )
    with pytest.raises(RuntimeError, match=match):
        cls.upgrade_checkpoint(checkpoint)
