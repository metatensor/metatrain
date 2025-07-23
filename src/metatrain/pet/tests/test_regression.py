import os
import random
from copy import deepcopy

import numpy as np
import torch
from metatomic.torch import ModelOutput
from omegaconf import OmegaConf

from metatrain.pet import PET, Trainer
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import (
    read_systems,
    read_targets,
)
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
from metatrain.utils.omegaconf import CONF_LOSS

from . import DATASET_PATH, DATASET_WITH_FORCES_PATH, DEFAULT_HYPERS, MODEL_HYPERS


def test_regression_init():
    """Regression test for the model at initialization"""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    os.environ["PYTHONHASHSEED"] = str(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    targets = {}
    targets["mtt::U0"] = get_energy_target_info({"quantity": "energy", "unit": "eV"})

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=targets
    )
    model = PET(MODEL_HYPERS, dataset_info)

    # Predict on the first five systems
    systems = read_systems(DATASET_PATH)[:5]
    systems = [system.to(torch.float32) for system in systems]
    for system in systems:
        get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

    output = model(
        systems,
        {"mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=False)},
    )

    expected_output = torch.tensor(
        [
            [3.015289306641],
            [1.845376491547],
            [0.776397109032],
            [1.858697414398],
            [1.014655113220],
        ]
    )

    # if you need to change the hardcoded values:
    # torch.set_printoptions(precision=12)
    # print(output["mtt::U0"].block().values)

    torch.testing.assert_close(output["mtt::U0"].block().values, expected_output)


def test_regression_energies_forces_train():
    """Regression test for the model when trained for 2 epoch on a small dataset"""

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.backends.cudnn.benchmark = False

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    os.environ["PYTHONHASHSEED"] = str(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    systems = read_systems(DATASET_WITH_FORCES_PATH)

    conf = {
        "energy": {
            "quantity": "energy",
            "read_from": DATASET_WITH_FORCES_PATH,
            "reader": "ase",
            "key": "energy",
            "unit": "eV",
            "type": "scalar",
            "per_atom": False,
            "num_subtargets": 1,
            "forces": {"read_from": DATASET_WITH_FORCES_PATH, "key": "force"},
            "stress": False,
            "virial": False,
        }
    }

    targets, target_info_dict = read_targets(OmegaConf.create(conf))
    targets = {"energy": targets["energy"]}
    dataset = Dataset.from_dict({"system": systems, "energy": targets["energy"]})
    hypers = deepcopy(DEFAULT_HYPERS)
    hypers["training"]["num_epochs"] = 2
    hypers["training"]["scheduler_patience"] = 1
    hypers["training"]["fixed_composition_weights"] = {}
    loss_conf = {"energy": deepcopy(CONF_LOSS)}
    loss_conf["energy"]["gradients"] = {"positions": deepcopy(CONF_LOSS)}
    loss_conf = OmegaConf.create(loss_conf)
    OmegaConf.resolve(loss_conf)
    hypers["training"]["loss"] = loss_conf

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[6], targets=target_info_dict
    )
    print("MODEL HYPERS:", MODEL_HYPERS)
    print("DATASET INFO:", dataset_info)
    print("CONTENT:", os.listdir("."))
    model = PET(MODEL_HYPERS, dataset_info)
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
    for system in systems:
        get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

    output = evaluate_model(
        model, systems[:5], targets=target_info_dict, is_training=False
    )

    expected_output = torch.tensor(
        [
            [20.386034011841],
            [20.353490829468],
            [20.303865432739],
            [20.413286209106],
            [20.318788528442],
        ]
    )

    expected_gradients_output = torch.tensor(
        [0.208536088467, -0.117365449667, -0.278660595417]
    )

    # if you need to change the hardcoded values:
    torch.set_printoptions(precision=12)
    print(output["energy"].block().values)
    print(output["energy"].block().gradient("positions").values.squeeze(-1)[0])

    raise

    torch.testing.assert_close(output["energy"].block().values, expected_output)
    torch.testing.assert_close(
        output["energy"].block().gradient("positions").values[0, :, 0],
        expected_gradients_output,
    )
