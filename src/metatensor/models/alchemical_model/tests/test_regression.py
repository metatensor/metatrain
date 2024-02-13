import random

import ase.io
import numpy as np
import rascaline.torch
import torch
from metatensor.learn.data import Dataset
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput
from omegaconf import OmegaConf

from metatensor.models.alchemical_model import DEFAULT_HYPERS, Model, train
from metatensor.models.alchemical_model.utils import get_primitive_neighbors_list
from metatensor.models.utils.data import get_all_species
from metatensor.models.utils.data.readers import read_structures, read_targets

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
    systems = []
    for structure in structures:
        nl, nl_options = get_primitive_neighbors_list(structure)
        system = rascaline.torch.systems_to_torch(structure)
        system.add_neighbors_list(nl_options, nl)
        systems.append(system)

    output = alchemical_model(
        systems,
        {"U0": alchemical_model.capabilities.outputs["U0"]},
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

    atoms_list = ase.io.read(DATASET_PATH, ":")
    structures = read_structures(DATASET_PATH)

    for atoms, structure in zip(atoms_list, structures):
        nl, nl_options = get_primitive_neighbors_list(
            atoms, model_cutoff=5.0, full_list=True
        )
        structure.add_neighbors_list(nl_options, nl)

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

    dataset = Dataset(structures, targets)

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
    output = alchemical_model(
        structures[:5], {"U0": alchemical_model.capabilities.outputs["U0"]}
    )

    expected_output = torch.tensor(
        [[-37.3165], [-33.7774], [-29.9353], [-61.5583], [-61.0081]],
        dtype=torch.float64,
    )

    assert torch.allclose(output["U0"].block().values, expected_output, rtol=1e-3)
