import ase
import rascaline.torch
import torch
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput

from metatensor.models.alchemical_model import DEFAULT_HYPERS, Model

from ..utils import get_primitive_neighbors_list


def test_prediction_subset():
    """Tests that the model can predict on a subset
    of the elements it was trained on."""

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        species=[1, 6, 7, 8],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
    )

    alchemical_model = Model(capabilities, DEFAULT_HYPERS["model"]).to(torch.float64)
    structure = ase.Atoms("O2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    nl, nl_options = get_primitive_neighbors_list(structure)
    system = rascaline.torch.systems_to_torch(structure)
    system.add_neighbors_list(nl_options, nl)

    alchemical_model(
        [system],
        {"energy": alchemical_model.capabilities.outputs["energy"]},
    )
