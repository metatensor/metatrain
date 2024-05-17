import shutil

import pytest
import torch
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput
from omegaconf import OmegaConf

import metatensor.models
from metatensor.models.experimental.soap_bpnn import DEFAULT_HYPERS, Model, train
from metatensor.models.utils.data import Dataset, DatasetInfo, TargetInfo
from metatensor.models.utils.data.readers import read_systems, read_targets
from metatensor.models.utils.io import export, save

from . import DATASET_PATH


def test_continue(monkeypatch, tmp_path):
    """Tests that a model can be checkpointed and loaded
    for a continuation of the training process"""

    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH, "qm9_reduced_100.xyz")

    systems = read_systems(DATASET_PATH)

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        outputs={
            "mtm::U0": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
        interaction_range=DEFAULT_HYPERS["model"]["soap"]["cutoff"],
        dtype="float32",
    )
    model_before = Model(capabilities, DEFAULT_HYPERS["model"])
    output_before = model_before(
        systems[:5], {"mtm::U0": model_before.capabilities.outputs["mtm::U0"]}
    )

    save(model_before, "model.ckpt")

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

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        targets={
            "mtm::U0": TargetInfo(
                quantity="energy",
                unit="eV",
            ),
        },
    )
    model_after = train(
        [dataset],
        [dataset],
        dataset_info,
        [torch.device("cpu")],
        hypers,
        continue_from="model.ckpt",
    )

    # Predict on the first five systems
    output_after = model_after(
        systems[:5], {"mtm::U0": model_after.capabilities.outputs["mtm::U0"]}
    )

    assert metatensor.torch.allclose(output_before["mtm::U0"], output_after["mtm::U0"])

    # test error raise of model is already exported
    export(model_before, "exported.pt")
    with pytest.raises(
        ValueError, match="model is already exported and can't be used for continue"
    ):
        train(
            [dataset],
            [dataset],
            dataset_info,
            [torch.device("cpu")],
            hypers,
            continue_from="exported.pt",
        )
