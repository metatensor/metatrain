import copy
import shutil

import metatensor
import torch
from omegaconf import OmegaConf

from metatrain.experimental.soap_bpnn import SoapBpnn, Trainer
from metatrain.utils.data import Dataset, DatasetInfo, TargetInfo, TargetInfoDict
from metatrain.utils.data.readers import read_systems, read_targets

from . import DATASET_PATH, DEFAULT_HYPERS, MODEL_HYPERS


def test_continue(monkeypatch, tmp_path):
    """Tests that a model can be checkpointed and loaded
    for a continuation of the training process"""

    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH, "qm9_reduced_100.xyz")

    systems = read_systems(DATASET_PATH)

    target_info_dict = TargetInfoDict()
    target_info_dict["mtt::U0"] = TargetInfo(quantity="energy", unit="eV")

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types={1, 6, 7, 8}, targets=target_info_dict
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
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, _ = read_targets(OmegaConf.create(conf))
    dataset = Dataset({"system": systems, "mtt::U0": targets["mtt::U0"]})

    hypers = DEFAULT_HYPERS.copy()
    hypers["training"]["num_epochs"] = 0

    model_before = copy.deepcopy(model)
    model_after = model.restart(dataset_info)

    hypers["training"]["num_epochs"] = 0
    trainer = Trainer(hypers["training"])
    trainer.train(model_after, [torch.device("cpu")], [dataset], [dataset], ".")

    # Predict on the first five systems
    output_before = model_before(
        systems[:5], {"mtt::U0": model_before.outputs["mtt::U0"]}
    )
    output_after = model_after(systems[:5], {"mtt::U0": model_after.outputs["mtt::U0"]})

    assert metatensor.torch.allclose(output_before["mtt::U0"], output_after["mtt::U0"])
