import copy
import shutil

import metatensor
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from omegaconf import OmegaConf

from metatrain.experimental.flashmd import FlashMD, Trainer
from metatrain.utils.data import Dataset, DatasetInfo, TargetInfo
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.io import model_from_checkpoint
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)
from metatrain.utils.omegaconf import CONF_LOSS

from . import DATASET_PATH, DEFAULT_HYPERS, MODEL_HYPERS


DEFAULT_HYPERS = copy.deepcopy(DEFAULT_HYPERS)
DEFAULT_HYPERS["training"]["timestep"] = 30.0
DEFAULT_HYPERS["training"]["batch_size"] = 2


@pytest.mark.filterwarnings("ignore:custom data:UserWarning")
def test_continue(monkeypatch, tmp_path):
    """Tests that a model can be checkpointed and loaded
    for a continuation of the training process"""

    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH, "flashmd.xyz")

    systems = read_systems("flashmd.xyz")

    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[13],
        targets={
            name: TargetInfo(
                layout=TensorMap(
                    keys=Labels.single(),
                    blocks=[
                        TensorBlock(
                            values=torch.empty((0, 3, 1), dtype=torch.float64),
                            samples=Labels(
                                names=["system", "atom"],
                                values=torch.empty((0, 2), dtype=int),
                            ),
                            components=[
                                Labels.range("xyz", 3),
                            ],
                            properties=Labels.range("length", 1),
                        )
                    ],
                ),
                quantity="length",
                unit="angstrom",
            )
            for name in ["positions", "momenta"]
        },
    )

    model = FlashMD(MODEL_HYPERS, dataset_info)
    requested_neighbor_lists = get_requested_neighbor_lists(model)
    systems = [
        get_system_with_neighbor_lists(system, requested_neighbor_lists)
        for system in systems
    ]

    output_before = model(
        [system.to(torch.float32) for system in systems[:5]],
        {"positions": model.outputs["positions"], "momenta": model.outputs["momenta"]},
    )

    positions_target = {
        "quantity": "position",
        "read_from": "flashmd.xyz",
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
        "read_from": "flashmd.xyz",
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

    conf = {
        "positions": positions_target,
        "momenta": momenta_target,
    }
    targets, _ = read_targets(OmegaConf.create(conf))

    dataset = Dataset.from_dict(
        {
            "system": systems,
            "positions": targets["positions"],
            "momenta": targets["momenta"],
        }
    )

    hypers = DEFAULT_HYPERS.copy()
    hypers["training"]["num_epochs"] = 0
    loss_conf = OmegaConf.create(
        {"positions": CONF_LOSS.copy(), "momenta": CONF_LOSS.copy()}
    )
    OmegaConf.resolve(loss_conf)
    hypers["training"]["loss"] = loss_conf

    trainer = Trainer(hypers["training"])
    trainer.train(
        model=model,
        dtype=torch.float32,
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir=".",
    )

    trainer.save_checkpoint(model, "temp.ckpt")
    checkpoint = torch.load("temp.ckpt", weights_only=False, map_location="cpu")
    model_after = model_from_checkpoint(checkpoint, context="restart")
    assert isinstance(model_after, FlashMD)
    model_after.restart(dataset_info)

    hypers["training"]["num_epochs"] = 0
    trainer = Trainer(hypers["training"])
    trainer.train(
        model=model_after,
        dtype=torch.float32,
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir=".",
    )

    # evaluation
    systems = [system.to(torch.float32) for system in systems]

    model.eval()
    model_after.eval()

    # Predict on the first five systems
    output_before = model(
        [system.to(torch.float32) for system in systems[:5]],
        {"positions": model.outputs["positions"], "momenta": model.outputs["momenta"]},
    )
    output_after = model_after(
        [system.to(torch.float32) for system in systems[:5]],
        {
            "positions": model_after.outputs["positions"],
            "momenta": model_after.outputs["momenta"],
        },
    )

    assert metatensor.torch.allclose(
        output_before["positions"], output_after["positions"]
    )
    assert metatensor.torch.allclose(output_before["momenta"], output_after["momenta"])
