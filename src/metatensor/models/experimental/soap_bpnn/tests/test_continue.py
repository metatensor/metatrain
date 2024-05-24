import copy
import shutil

import torch
from omegaconf import OmegaConf

import metatensor.models
from metatensor.models.experimental.soap_bpnn import SOAPBPNN, Trainer
from metatensor.models.utils.data import Dataset, DatasetInfo, TargetInfo
from metatensor.models.utils.data.readers import read_systems, read_targets

from . import DATASET_PATH, DEFAULT_HYPERS, MODEL_HYPERS


def test_continue(monkeypatch, tmp_path):
    """Tests that a model can be checkpointed and loaded
    for a continuation of the training process"""

    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH, "qm9_reduced_100.xyz")

    systems = read_systems(DATASET_PATH)

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "mtm::U0": TargetInfo(
                quantity="energy",
                unit="eV",
            )
        },
    )
    model = SOAPBPNN(MODEL_HYPERS, dataset_info)
    output_before = model(systems[:5], {"mtm::U0": model.outputs["mtm::U0"]})

    conf = {
        "mtm::U0": {
            "quantity": "energy",
            "read_from": DATASET_PATH,
            "file_format": ".xyz",
            "key": "U0",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets = read_targets(OmegaConf.create(conf))
    dataset = Dataset({"system": systems, "mtm::U0": targets["mtm::U0"]})

    hypers = DEFAULT_HYPERS.copy()
    hypers["training"]["num_epochs"] = 0

    model_before = copy.deepcopy(model)
    model_after = model.restart(dataset_info)

    hypers["training"]["num_epochs"] = 0
    trainer = Trainer(hypers["training"])
    trainer.train(model_after, [torch.device("cpu")], [dataset], [dataset], ".")

    # Predict on the first five systems
    output_before = model_before(
        systems[:5], {"mtm::U0": model_before.outputs["mtm::U0"]}
    )
    output_after = model_after(systems[:5], {"mtm::U0": model_after.outputs["mtm::U0"]})

    assert metatensor.torch.allclose(output_before["mtm::U0"], output_after["mtm::U0"])
