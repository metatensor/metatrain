import copy

import pytest
import torch

from metatrain.soap_bpnn import SoapBpnn, Trainer
from metatrain.utils.data import (
    DatasetInfo,
    get_atomic_types,
    get_dataset,
)
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.testing.checkpoints import (
    checkpoint_did_not_change,
    make_checkpoint_load_tests,
)

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
    hypers["soap"]["max_angular"] = 1
    hypers["soap"]["max_radial"] = 1
    hypers["bpnn"]["num_neurons_per_layer"] = 1
    hypers["bpnn"]["num_hidden_layers"] = 1

    model = SoapBpnn(hypers, dataset_info)

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


test_checkpoint_did_not_change = checkpoint_did_not_change

test_loading_old_checkpoints = make_checkpoint_load_tests(DEFAULT_HYPERS)


@pytest.mark.parametrize("context", ["finetune", "restart", "export"])
def test_get_checkpoint(context):
    """
    Test that the checkpoint created by the model.get_checkpoint()
    function can be loaded back in all possible contexts.
    """
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )
    model = SoapBpnn(MODEL_HYPERS, dataset_info)
    checkpoint = model.get_checkpoint()
    SoapBpnn.load_checkpoint(checkpoint, context)


def test_failed_model_checkpoint_upgrade():
    """Test that an error is raised when trying to upgrade an invalid checkpoint."""
    checkpoint = {"model_ckpt_version": 9999}

    match = (
        f"Unable to upgrade the checkpoint: the checkpoint is using "
        f"version 9999, while the current "
        f"version is {SoapBpnn.__checkpoint_version__}."
    )
    with pytest.raises(RuntimeError, match=match):
        SoapBpnn.upgrade_checkpoint(checkpoint)


def test_failed_trainer_checkpoint_upgrade():
    """Test that an error is raised when trying to upgrade an invalid checkpoint."""
    checkpoint = {"trainer_ckpt_version": 9999}

    match = (
        f"Unable to upgrade the checkpoint: the checkpoint is using "
        f"version 9999, while the current "
        f"version is {Trainer.__checkpoint_version__}."
    )
    with pytest.raises(RuntimeError, match=match):
        Trainer.upgrade_checkpoint(checkpoint)
