import shutil

import metatensor
import torch
from omegaconf import OmegaConf

from metatrain.pet import PET, Trainer
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.io import model_from_checkpoint
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
from metatrain.utils.omegaconf import CONF_LOSS

from . import DATASET_PATH, DEFAULT_HYPERS, MODEL_HYPERS


def test_continue(monkeypatch, tmp_path):
    """Tests that a model can be checkpointed and loaded
    for a continuation of the training process"""

    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH, "qm9_reduced_100.xyz")

    systems = read_systems(DATASET_PATH)

    target_info_dict = {}
    target_info_dict["mtt::U0"] = get_energy_target_info(
        {"quantity": "energy", "unit": "eV"}
    )

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=target_info_dict
    )
    model = PET(MODEL_HYPERS, dataset_info)

    conf = {
        "mtt::U0": {
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
    }
    targets, _ = read_targets(OmegaConf.create(conf))

    # systems in float64 are required for training
    systems = [system.to(torch.float64) for system in systems]

    dataset = Dataset.from_dict({"system": systems, "mtt::U0": targets["mtt::U0"]})

    hypers = DEFAULT_HYPERS.copy()
    hypers["training"]["num_epochs"] = 0
    loss_conf = OmegaConf.create({"mtt::U0": CONF_LOSS.copy()})
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

    trainer.save_checkpoint(model, "tmp.ckpt")

    model_after = model_from_checkpoint("tmp.ckpt", context="restart")
    assert isinstance(model_after, PET)
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
    systems = read_systems(DATASET_PATH)
    systems = [system.to(torch.float32) for system in systems[:5]]
    for system in systems:
        system.positions.requires_grad_(True)
        get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

    model.eval()
    model_after.eval()

    output_before = model(systems[:5], {"mtt::U0": model.outputs["mtt::U0"]})
    output_after = model_after(systems[:5], {"mtt::U0": model_after.outputs["mtt::U0"]})

    assert metatensor.torch.allclose(output_before["mtt::U0"], output_after["mtt::U0"])

    # also check forces predictions
    energy_before = output_before["mtt::U0"].block().values
    energy_before.backward(torch.ones_like(energy_before))

    forces_before = [s.positions.grad for s in systems]

    for system in systems:
        system.positions.grad = None

    energy_after = output_after["mtt::U0"].block().values
    energy_after.backward(torch.ones_like(energy_after))

    forces_after = [s.positions.grad for s in systems]

    assert torch.allclose(torch.vstack(forces_before), torch.vstack(forces_after))
