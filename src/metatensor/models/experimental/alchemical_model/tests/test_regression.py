import random

import ase.io
import numpy as np
import rascaline.torch
import torch
from metatensor.learn.data import Dataset
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
)
from omegaconf import OmegaConf

from metatensor.models.experimental.alchemical_model import DEFAULT_HYPERS, Model, train
from metatensor.models.utils.data import get_all_species
from metatensor.models.utils.data.readers import read_systems, read_targets
from metatensor.models.utils.neighbors_lists import get_system_with_neighbors_lists

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
    systems = [
        rascaline.torch.systems_to_torch(system).to(torch.get_default_dtype())
        for system in systems
    ]
    systems = [
        get_system_with_neighbors_lists(
            system, alchemical_model.requested_neighbors_lists()
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

    expected_output = torch.tensor(
        [[-1.1830e-03], [-1.7822e-03], [7.2585e-06], [-5.2279e-04], [-4.1751e-04]]
    )

    assert torch.allclose(output["U0"].block().values, expected_output, rtol=1e-3)


def test_regression_train():
    """Perform a regression test on the model when
    trained for 2 epoch on a small dataset"""

    # reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    systems = read_systems(DATASET_PATH, dtype=torch.get_default_dtype())
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
    targets = read_targets(OmegaConf.create(conf), dtype=torch.get_default_dtype())
    dataset = Dataset(system=systems, U0=targets["U0"])

    hypers = DEFAULT_HYPERS.copy()
    hypers["training"]["num_epochs"] = 2

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        atomic_types=get_all_species(dataset),
        outputs={
            "U0": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
        supported_devices=["cpu"],
    )
    alchemical_model = train(
        train_datasets=[dataset],
        validation_datasets=[dataset],
        requested_capabilities=capabilities,
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
        [[-40.4883], [-56.5384], [-76.4003], [-77.3402], [-93.4341]]
    )

    assert torch.allclose(output["U0"].block().values, expected_output, rtol=1e-3)
