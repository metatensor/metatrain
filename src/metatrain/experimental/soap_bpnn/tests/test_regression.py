import random

import numpy as np
import torch
from metatensor.torch.atomistic import ModelOutput
from omegaconf import OmegaConf

from metatrain.experimental.soap_bpnn import SoapBpnn, Trainer
from metatrain.utils.data import Dataset, DatasetInfo, TargetInfo, TargetInfoDict
from metatrain.utils.data.readers import read_systems, read_targets

from . import DATASET_PATH, DEFAULT_HYPERS, MODEL_HYPERS


# reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def test_regression_init():
    """Perform a regression test on the model at initialization"""

    targets = TargetInfoDict()
    targets["mtm::U0"] = TargetInfo(quantity="energy", unit="eV")

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types={1, 6, 7, 8}, targets=targets
    )
    model = SoapBpnn(MODEL_HYPERS, dataset_info)

    # Predict on the first five systems
    systems = read_systems(DATASET_PATH)[:5]

    output = model(
        systems,
        {"mtm::U0": ModelOutput(quantity="energy", unit="", per_atom=False)},
    )

    expected_output = torch.tensor(
        [[-0.03860], [0.11137], [0.09112], [-0.05634], [-0.02549]]
    )

    # if you need to change the hardcoded values:
    torch.set_printoptions(precision=5)
    print(output["mtm::U0"].block().values)

    torch.testing.assert_close(
        output["mtm::U0"].block().values, expected_output, rtol=1e-5, atol=1e-5
    )


def test_regression_train():
    """Perform a regression test on the model when
    trained for 2 epoch on a small dataset"""

    systems = read_systems(DATASET_PATH)

    conf = {
        "mtm::U0": {
            "quantity": "energy",
            "read_from": DATASET_PATH,
            "file_format": ".xyz",
            "key": "U0",
            "unit": "eV",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, target_info_dict = read_targets(OmegaConf.create(conf))
    dataset = Dataset({"system": systems, "mtm::U0": targets["mtm::U0"]})

    hypers = DEFAULT_HYPERS.copy()
    hypers["training"]["num_epochs"] = 2

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types={1, 6, 7, 8}, targets=target_info_dict
    )
    model = SoapBpnn(MODEL_HYPERS, dataset_info)

    hypers["training"]["num_epochs"] = 1
    trainer = Trainer(hypers["training"])
    trainer.train(model, [torch.device("cpu")], [dataset], [dataset], ".")

    # Predict on the first five systems
    output = model(
        systems[:5],
        {"mtm::U0": ModelOutput(quantity="energy", unit="", per_atom=False)},
    )

    expected_output = torch.tensor(
        [[-40.56458], [-56.51794], [-76.49743], [-77.32737], [-93.40791]]
    )

    # if you need to change the hardcoded values:
    # torch.set_printoptions(precision=5)
    # print(output["mtm::U0"].block().values)

    torch.testing.assert_close(
        output["mtm::U0"].block().values, expected_output, rtol=1e-5, atol=1e-5
    )
