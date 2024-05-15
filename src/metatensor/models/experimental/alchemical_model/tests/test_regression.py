import random

import ase.io
import numpy as np
import torch
from metatensor.learn.data import Dataset
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    systems_to_torch,
)
from omegaconf import OmegaConf

from metatensor.models.experimental.alchemical_model import DEFAULT_HYPERS, Model, train
from metatensor.models.utils.data import DatasetInfo, TargetInfo
from metatensor.models.utils.data.readers import read_systems, read_targets
from metatensor.models.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DATASET_PATH


def test_regression_init():
    """Perform a regression test on the model at initialization"""

    # reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        outputs={
            "U0": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
        supported_devices=["cpu"],
    )
    alchemical_model = Model(capabilities, DEFAULT_HYPERS["model"])

    # Predict on the first five systems
    systems = ase.io.read(DATASET_PATH, ":5")
    systems = [systems_to_torch(system) for system in systems]
    systems = [
        get_system_with_neighbor_lists(
            system, alchemical_model.requested_neighbor_lists()
        )
        for system in systems
    ]

    evaluation_options = ModelEvaluationOptions(
        length_unit=capabilities.length_unit,
        outputs=capabilities.outputs,
    )

    model = MetatensorAtomisticModel(
        alchemical_model.eval(), ModelMetadata(), alchemical_model.capabilities
    )
    output = model(
        systems,
        evaluation_options,
        check_consistency=True,
    )

    expected_output = torch.tensor([[-1.9819], [0.1507], [1.6116], [3.4118], [0.8383]])

    torch.testing.assert_close(
        output["U0"].block().values, expected_output, rtol=1e-05, atol=1e-4
    )


def test_regression_train():
    """Perform a regression test on the model when
    trained for 2 epoch on a small dataset"""

    # reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    systems = read_systems(DATASET_PATH)
    conf = {
        "U0": {
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
    dataset = Dataset(system=systems, U0=targets["U0"])

    hypers = DEFAULT_HYPERS.copy()
    hypers["training"]["num_epochs"] = 2

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        targets={
            "U0": TargetInfo(
                quantity="energy",
                unit="eV",
            ),
        },
    )
    alchemical_model = train(
        train_datasets=[dataset],
        validation_datasets=[dataset],
        dataset_info=dataset_info,
        devices=[torch.device("cpu")],
        hypers=hypers,
    )

    # Predict on the first five systems
    evaluation_options = ModelEvaluationOptions(
        length_unit=alchemical_model.capabilities.length_unit,
        outputs=alchemical_model.capabilities.outputs,
    )

    model = MetatensorAtomisticModel(
        alchemical_model.eval(), ModelMetadata(), alchemical_model.capabilities
    )
    output = model(
        systems[:5],
        evaluation_options,
        check_consistency=True,
    )

    expected_output = torch.tensor(
        [[-126.6899], [-113.0781], [-135.8210], [-179.1740], [-149.5980]]
    )

    torch.testing.assert_close(
        output["U0"].block().values, expected_output, rtol=1e-05, atol=1e-4
    )
