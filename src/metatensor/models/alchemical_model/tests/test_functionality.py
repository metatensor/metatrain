import ase
import rascaline.torch
import torch
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput

from metatensor.models.alchemical_model import DEFAULT_HYPERS, Model
from metatensor.models.utils.neighbors_lists import get_rascaline_neighbors_list


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
    system = rascaline.torch.systems_to_torch(structure)
    requested_neighbors_lists = alchemical_model.requested_neighbors_lists()
    for nl_options in requested_neighbors_lists:
        nl = get_rascaline_neighbors_list(system, nl_options)
        system.add_neighbors_list(nl_options, nl)

    alchemical_model(
        [system],
        {"energy": alchemical_model.capabilities.outputs["energy"]},
    )
