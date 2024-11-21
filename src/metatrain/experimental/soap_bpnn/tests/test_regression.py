import random

import numpy as np
import torch
from metatensor.torch.atomistic import ModelOutput
from omegaconf import OmegaConf

from metatrain.experimental.soap_bpnn import SoapBpnn, Trainer
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.data.target_info import get_energy_target_info

from . import DATASET_PATH, DEFAULT_HYPERS, MODEL_HYPERS


# reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def test_regression_init():
    """Perform a regression test on the model at initialization"""

    targets = {}
    targets["mtt::U0"] = get_energy_target_info({"unit": "eV"})

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=targets
    )
    model = SoapBpnn(MODEL_HYPERS, dataset_info)

    # Predict on the first five systems
    systems = read_systems(DATASET_PATH)[:5]
    systems = [system.to(torch.float32) for system in systems]

    output = model(
        systems,
        {"mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=False)},
    )

    expected_output = torch.tensor(
        [
            [-0.038599025458],
            [0.111374437809],
            [0.091115802526],
            [-0.056339077652],
            [-0.025491207838],
        ]
    )

    # if you need to change the hardcoded values:
    torch.set_printoptions(precision=12)
    print(output["mtt::U0"].block().values)

    torch.testing.assert_close(
        output["mtt::U0"].block().values, expected_output, rtol=1e-5, atol=1e-5
    )


def test_regression_train():
    """Perform a regression test on the model when
    trained for 2 epoch on a small dataset"""

    systems = read_systems(DATASET_PATH)

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
    targets, target_info_dict = read_targets(OmegaConf.create(conf))
    dataset = Dataset.from_dict({"system": systems, "mtt::U0": targets["mtt::U0"]})

    hypers = DEFAULT_HYPERS.copy()
    hypers["training"]["num_epochs"] = 2

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=target_info_dict
    )
    model = SoapBpnn(MODEL_HYPERS, dataset_info)

    hypers["training"]["num_epochs"] = 1
    trainer = Trainer(hypers["training"])
    trainer.train(
        model=model,
        dtype=torch.float32,
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir=".",
    )

    # Predict on the first five systems
    systems = [system.to(torch.float32) for system in systems]
    output = model(
        systems[:5],
        {"mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=False)},
    )

    expected_output = torch.tensor(
        [
            [-0.106249026954],
            [0.039981484413],
            [-0.142682999372],
            [-0.031701669097],
            [-0.016210660338],
        ]
    )

    # if you need to change the hardcoded values:
    torch.set_printoptions(precision=12)
    print(output["mtt::U0"].block().values)

    torch.testing.assert_close(
        output["mtt::U0"].block().values, expected_output, rtol=1e-5, atol=1e-5
    )
