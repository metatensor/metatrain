import random

import numpy as np
import torch
from metatomic.torch import ModelOutput
from omegaconf import OmegaConf

from metatrain.experimental.phace import PhACE, Trainer
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DATASET_PATH, DEFAULT_HYPERS, MODEL_HYPERS


# reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def test_regression_init():
    """Perform a regression test on the model at initialization"""

    # Single thread ensures deterministic float accumulation in index_add_
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    targets = {}
    targets["mtt::U0"] = get_energy_target_info(
        "energy", {"quantity": "energy", "unit": "eV"}
    )

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=targets
    )
    model = PhACE(MODEL_HYPERS, dataset_info)

    model.module = model.fake_gradient_model
    del model.gradient_model
    del model.fake_gradient_model
    model = torch.jit.script(model)

    # Predict on the first five systems
    systems = read_systems(DATASET_PATH)[:5]
    systems = [system.to(torch.float32) for system in systems]
    for system in systems:
        get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

    output = model(
        systems,
        {"mtt::U0": ModelOutput(quantity="energy", unit="", sample_kind="system")},
    )

    expected_output = torch.tensor(
        [
            [53.651760101318],
            [31.206809997559],
            [31.099349975586],
            [43.266471862793],
            [23.615173339844],
        ]
    )

    # if you need to change the hardcoded values:
    # torch.set_printoptions(precision=12)
    # print(output["mtt::U0"].block().values)

    torch.testing.assert_close(
        output["mtt::U0"].block().values, expected_output, rtol=1e-5, atol=1e-5
    )

    torch.set_num_threads(n_threads)


def test_regression_train():
    """Perform a regression test on the model when
    trained for 2 epoch on a small dataset"""

    # Single thread ensures deterministic float accumulation in index_add_
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    systems = read_systems(DATASET_PATH)

    conf = {
        "mtt::U0": {
            "quantity": "energy",
            "read_from": DATASET_PATH,
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, target_info_dict = read_targets(OmegaConf.create(conf))
    dataset = Dataset.from_dict({"system": systems, "mtt::U0": targets["mtt::U0"]})

    hypers = DEFAULT_HYPERS.copy()
    hypers["training"]["scheduler_patience"] = 1
    hypers["training"]["fixed_composition_weights"] = {}
    loss_conf = {"energy": init_with_defaults(LossSpecification)}
    loss_conf["energy"]["gradients"] = {
        "positions": init_with_defaults(LossSpecification)
    }
    loss_conf = OmegaConf.create(loss_conf)
    OmegaConf.resolve(loss_conf)
    hypers["training"]["loss"] = loss_conf
    hypers["training"]["num_epochs"] = 2
    hypers["training"]["num_workers"] = 0  # for reproducibility
    hypers["training"]["compile"] = False

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=target_info_dict
    )
    model = PhACE(MODEL_HYPERS, dataset_info)

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
    for system in systems:
        get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

    model.module = model.fake_gradient_model
    del model.gradient_model
    del model.fake_gradient_model
    model = torch.jit.script(model)
    output = model(
        systems[:5],
        {"mtt::U0": ModelOutput(quantity="energy", unit="", sample_kind="system")},
    )

    expected_output = torch.tensor(
        [
            [12.195680618286],
            [29.831180572510],
            [25.561126708984],
            [11.692634582520],
            [23.836612701416],
        ]
    )

    # if you need to change the hardcoded values:
    # torch.set_printoptions(precision=12)
    # print(output["mtt::U0"].block().values)

    # Training amplifies cross-hardware float differences (different SIMD paths,
    # math libraries) through backprop + optimizer steps. Single-threaded execution
    # makes results deterministic on a given machine, but CI runners differ from
    # local machines by up to ~0.02 absolute.
    torch.testing.assert_close(
        output["mtt::U0"].block().values, expected_output, rtol=5e-3, atol=0.05
    )

    torch.set_num_threads(n_threads)
