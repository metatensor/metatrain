import shutil

import metatensor
import torch
from omegaconf import OmegaConf

from metatrain.soap_bpnn import SoapBpnn, Trainer
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.data.target_info import get_energy_target_info

from . import DATASET_PATH, DEFAULT_HYPERS, MODEL_HYPERS


def test_continue(monkeypatch, tmp_path):
    """Tests that a model can be checkpointed and loaded
    for a continuation of the training process"""

    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH, "qm9_reduced_100.xyz")

    systems = read_systems(DATASET_PATH)
    systems = [system.to(torch.float32) for system in systems]

    target_info_dict = {}
    target_info_dict["mtt::U0"] = get_energy_target_info({"unit": "eV"})

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=target_info_dict
    )
    model = SoapBpnn(MODEL_HYPERS, dataset_info)
    output_before = model(systems[:5], {"mtt::U0": model.outputs["mtt::U0"]})

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
    model_after = SoapBpnn.load_checkpoint("temp.ckpt")
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
    output_before = model(systems[:5], {"mtt::U0": model.outputs["mtt::U0"]})
    output_after = model_after(systems[:5], {"mtt::U0": model_after.outputs["mtt::U0"]})

    assert metatensor.torch.allclose(output_before["mtt::U0"], output_after["mtt::U0"])
