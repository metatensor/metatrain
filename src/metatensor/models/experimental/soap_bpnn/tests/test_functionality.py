import ase
import metatensor.torch
import torch
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput, systems_to_torch

from metatensor.models.experimental.soap_bpnn import DEFAULT_HYPERS, Model


def test_prediction_subset_elements():
    """Tests that the model can predict on a subset
    of the elements it was trained on."""

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
        interaction_range=DEFAULT_HYPERS["model"]["soap"]["cutoff"],
        dtype="float32",
    )

    soap_bpnn = Model(capabilities, DEFAULT_HYPERS["model"])

    system = ase.Atoms("O2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    soap_bpnn(
        [systems_to_torch(system)],
        {"energy": soap_bpnn.capabilities.outputs["energy"]},
    )


def test_prediction_subset_atoms():
    """Tests that the model can predict on a subset
    of the atoms in a system."""

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
        interaction_range=DEFAULT_HYPERS["model"]["soap"]["cutoff"],
        dtype="float32",
    )

    soap_bpnn = Model(capabilities, DEFAULT_HYPERS["model"])

    # Since we don't yet support atomic predictions, we will test this by
    # predicting on a system with two monomers at a large distance

    system_monomer = ase.Atoms(
        "NO2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]
    )

    energy_monomer = soap_bpnn(
        [systems_to_torch(system_monomer)],
        {"energy": soap_bpnn.capabilities.outputs["energy"]},
    )

    system_far_away_dimer = ase.Atoms(
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
        [systems_to_torch(system_far_away_dimer)],
        {"energy": soap_bpnn.capabilities.outputs["energy"]},
    )

    energy_monomer_in_dimer = soap_bpnn(
        [systems_to_torch(system_far_away_dimer)],
        {"energy": soap_bpnn.capabilities.outputs["energy"]},
        selected_atoms=selection_labels,
    )

    assert not metatensor.torch.allclose(
        energy_monomer["energy"], energy_dimer["energy"]
    )
    assert metatensor.torch.allclose(
        energy_monomer["energy"], energy_monomer_in_dimer["energy"]
    )
