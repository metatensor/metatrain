import ase
import metatensor.torch
import rascaline.torch
import torch
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput

from metatensor.models.experimental.soap_bpnn import DEFAULT_HYPERS, Model


def test_prediction_subset_elements():
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

    soap_bpnn = Model(capabilities, DEFAULT_HYPERS["model"])

    structure = ase.Atoms("O2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    soap_bpnn(
        [rascaline.torch.systems_to_torch(structure).to(torch.get_default_dtype())],
        {"energy": soap_bpnn.capabilities.outputs["energy"]},
    )


def test_prediction_subset_atoms():
    """Tests that the model can predict on a subset
    of the atoms in a structure."""

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

    soap_bpnn = Model(capabilities, DEFAULT_HYPERS["model"])

    # Since we don't yet support atomic predictions, we will test this by
    # predicting on a structure with two monomers at a large distance

    structure_monomer = ase.Atoms(
        "NO2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]
    )

    energy_monomer = soap_bpnn(
        [
            rascaline.torch.systems_to_torch(structure_monomer).to(
                torch.get_default_dtype()
            )
        ],
        {"energy": soap_bpnn.capabilities.outputs["energy"]},
    )

    structure_far_away_dimer = ase.Atoms(
        "N2O4",
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 50.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
            [0.0, 51.0, 0.0],
            [0.0, 42.0, 0.0],
        ],
    )

    selection_labels = metatensor.torch.Labels(
        names=["system", "atom"],
        values=torch.tensor([[0, 0], [0, 2], [0, 3]]),
    )

    energy_dimer = soap_bpnn(
        [
            rascaline.torch.systems_to_torch(structure_far_away_dimer).to(
                torch.get_default_dtype()
            )
        ],
        {"energy": soap_bpnn.capabilities.outputs["energy"]},
    )

    energy_monomer_in_dimer = soap_bpnn(
        [
            rascaline.torch.systems_to_torch(structure_far_away_dimer).to(
                torch.get_default_dtype()
            )
        ],
        {"energy": soap_bpnn.capabilities.outputs["energy"]},
        selected_atoms=selection_labels,
    )

    assert not metatensor.torch.allclose(
        energy_monomer["energy"], energy_dimer["energy"]
    )
    assert metatensor.torch.allclose(
        energy_monomer["energy"], energy_monomer_in_dimer["energy"]
    )
