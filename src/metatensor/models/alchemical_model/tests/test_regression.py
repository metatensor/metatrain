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
    ModelOutput,
)
from omegaconf import OmegaConf

from metatensor.models.alchemical_model import DEFAULT_HYPERS, Model, train
from metatensor.models.utils.data import get_all_species
from metatensor.models.utils.data.readers import read_structures, read_targets
from metatensor.models.alchemical_model.utils.neighbors_lists import (
    get_rascaline_neighbors_list,
)

from . import DATASET_PATH


def test_regression_init():
    """Perform a regression test on the model at initialization"""

    # reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        species=[1, 6, 7, 8],
        outputs={
            "U0": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
    )
    alchemical_model = Model(capabilities, DEFAULT_HYPERS["model"]).to(torch.float64)

    # Predict on the first fivestructures
    structures = ase.io.read(DATASET_PATH, ":5")
    systems = [rascaline.torch.systems_to_torch(structure) for structure in structures]
    requested_neighbors_lists = alchemical_model.requested_neighbors_lists()
    for nl_options in requested_neighbors_lists:
        nls = get_rascaline_neighbors_list(systems, nl_options)
        for system, nl in zip(systems, nls):
            system.add_neighbors_list(nl_options, nl)

    evaluation_options = ModelEvaluationOptions(
        length_unit=capabilities.length_unit,
        outputs=capabilities.outputs,
    )

    model = MetatensorAtomisticModel(
        alchemical_model.eval(), alchemical_model.capabilities
    )
    output = model(
        systems,
        evaluation_options,
        check_consistency=True,
    )
    expected_output = torch.tensor(
        [[-0.6996], [-0.4681], [2.2749], [-0.5971], [1.6994]],
        dtype=torch.float64,
    )

    assert torch.allclose(output["U0"].block().values, expected_output, rtol=1e-3)


def test_regression_train():
    """Perform a regression test on the model when
    trained for 2 epoch on a small dataset"""

    # reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    structures = read_structures(DATASET_PATH)
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
    dataset = Dataset(structure=structures, U0=targets["U0"])

    hypers = DEFAULT_HYPERS.copy()
    hypers["training"]["num_epochs"] = 2

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        species=get_all_species(dataset),
        outputs={
            "U0": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
    )
    alchemical_model = train(
        train_datasets=[dataset],
        validation_datasets=[dataset],
        requested_capabilities=capabilities,
        hypers=hypers,
    )

    # Predict on the first five structures
    evaluation_options = ModelEvaluationOptions(
        length_unit=alchemical_model.capabilities.length_unit,
        outputs=alchemical_model.capabilities.outputs,
    )

    model = MetatensorAtomisticModel(
        alchemical_model.eval(), alchemical_model.capabilities
    )
    output = model(
        structures[:5],
        evaluation_options,
        check_consistency=True,
    )

    expected_output = torch.tensor(
        [[-38.2606], [-37.4233], [-34.4073], [-58.7029], [-64.3037]],
        dtype=torch.float64,
    )

    assert torch.allclose(output["U0"].block().values, expected_output, rtol=1e-3)
