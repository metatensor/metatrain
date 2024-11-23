import random

import numpy as np
import torch
from metatensor.torch.atomistic import ModelEvaluationOptions
from omegaconf import OmegaConf

from metatrain.experimental.alchemical_model import AlchemicalModel, Trainer
from metatrain.utils.data import Dataset, DatasetInfo, read_systems, read_targets
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

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
    model = AlchemicalModel(MODEL_HYPERS, dataset_info)

    # Predict on the first five systems
    systems = read_systems(DATASET_PATH)[:5]
    requested_neighbor_lists = get_requested_neighbor_lists(model)
    systems = [
        get_system_with_neighbor_lists(system, requested_neighbor_lists)
        for system in systems
    ]

    evaluation_options = ModelEvaluationOptions(
        length_unit=model.dataset_info.length_unit,
        outputs=model.outputs,
    )

    exported = model.export()

    systems = [system.to(dtype=torch.float32) for system in systems]
    output = exported(systems, evaluation_options, check_consistency=True)

    expected_output = torch.tensor(
        [
            [-11.203639984131],
            [4.095238208771],
            [-4.632149219513],
            [-13.758152008057],
            [-2.430717945099],
        ],
    )

    # if you need to change the hardcoded values:
    # torch.set_printoptions(precision=12)
    # print(output["mtt::U0"].block().values)

    torch.testing.assert_close(
        output["mtt::U0"].block().values,
        expected_output,
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

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=target_info_dict
    )
    model = AlchemicalModel(MODEL_HYPERS, dataset_info)

    requested_neighbor_lists = get_requested_neighbor_lists(model)
    systems = [
        get_system_with_neighbor_lists(system, requested_neighbor_lists)
        for system in systems
    ]

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
    evaluation_options = ModelEvaluationOptions(
        length_unit=dataset_info.length_unit,
        outputs=model.outputs,
    )

    exported = model.export()

    systems = [system.to(dtype=torch.float32) for system in systems]
    output = exported(systems[:5], evaluation_options, check_consistency=True)

    expected_output = torch.tensor(
        [
            [-40.115474700928],
            [-56.302265167236],
            [-76.722442626953],
            [-77.022941589355],
            [-92.791801452637],
        ],
    )

    # if you need to change the hardcoded values:
    # torch.set_printoptions(precision=12)
    # print(output["mtt::U0"].block().values)

    torch.testing.assert_close(
        output["mtt::U0"].block().values,
        expected_output,
    )
