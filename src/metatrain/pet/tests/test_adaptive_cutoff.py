import ase
import pytest
import torch
from metatensor.torch import Labels
from metatomic.torch import systems_to_torch

from metatrain.pet import PET
from metatrain.pet.modules.structures import systems_to_batch
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import (
    get_energy_target_info,
)
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import MODEL_HYPERS


@pytest.fixture
def adaptive_cutoff_test_systems():
    """Creates a test system for adaptive cutoff testing."""
    isolated_atom = ase.Atoms("H", positions=[[0.0, 0.0, 0.0]])
    connected_dimer = ase.Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    disconnected_dimer = ase.Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 20.0]])
    connected_trimer = ase.Atoms(
        "H3", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74], [0.0, 0.0, 1.48]]
    )
    dimer_with_monomer = ase.Atoms(
        "H3", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74], [0.0, 0.0, 20.0]]
    )
    cluster = ase.Atoms(
        "H4",
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.74],
            [0.74, 0.0, 0.0],
            [0.0, 0.74, 0.0],
        ],
    )

    cluster_with_monomer = ase.Atoms(
        "H5",
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.74],
            [0.74, 0.0, 0.0],
            [0.0, 0.74, 0.0],
            [20.0, 20.0, 20.0],
        ],
    )

    two_clusters = ase.Atoms(
        "H8",
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.74],
            [0.74, 0.0, 0.0],
            [0.0, 0.74, 0.0],
            [5.0, 5.0, 5.0],
            [5.0, 5.0, 5.74],
            [5.74, 5.0, 5.0],
            [5.0, 5.74, 5.0],
        ],
    )

    chain_1d = ase.Atoms(
        "H",
        positions=[[0.0, 0.0, 0.0]],
        cell=[0.74, 0.0, 0.0],
        pbc=[True, False, False],
    )

    grid_2d = ase.Atoms(
        "H",
        positions=[[0.0, 0.0, 0.0]],
        cell=[0.74, 0.74, 0.0],
        pbc=[True, True, False],
    )

    grid_3d = ase.Atoms(
        "H",
        positions=[[0.0, 0.0, 0.0]],
        cell=[0.74, 0.74, 0.74],
        pbc=[True, True, True],
    )

    adaptive_cutoff_test_systems = [
        isolated_atom,
        connected_dimer,
        disconnected_dimer,
        connected_trimer,
        dimer_with_monomer,
        cluster,
        cluster_with_monomer,
        two_clusters,
        chain_1d,
        grid_2d,
        grid_3d,
    ]
    systems = systems_to_torch(adaptive_cutoff_test_systems)
    return systems


def test_adaptive_cutoff(adaptive_cutoff_test_systems):
    """Tests the adaptive cutoff functionality."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )

    model_hypers = MODEL_HYPERS.copy()
    model_hypers["cutoff"] = 7.0
    model_hypers["max_num_neighbors"] = 2
    model = PET(model_hypers, dataset_info)
    options = model.requested_neighbor_lists()[0]
    systems = [
        get_system_with_neighbor_lists(system, [options]).to(torch.float32)
        for system in adaptive_cutoff_test_systems
    ]

    systems_to_batch(
        systems,
        options,
        model.atomic_types,
        model.species_to_species_index,
        model.cutoff_width,
        max_num_neighbors=model.max_num_neighbors,
    )


@pytest.mark.parametrize("max_num_neighbors", [None, 2, 4, 8, 16, 32, 64])
def test_adaptive_cutoff_num_neighbors(
    monkeypatch, tmp_path, adaptive_cutoff_test_systems, max_num_neighbors
):
    """Tests the adaptive cutoff functionality."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )

    model_hypers = MODEL_HYPERS.copy()
    model_hypers["cutoff"] = 7.0
    model_hypers["max_num_neighbors"] = max_num_neighbors
    model = PET(model_hypers, dataset_info)
    options = model.requested_neighbor_lists()[0]
    systems = [
        get_system_with_neighbor_lists(system, [options]).to(torch.float32)
        for system in adaptive_cutoff_test_systems
    ]

    systems_to_batch(
        systems,
        options,
        model.atomic_types,
        model.species_to_species_index,
        model.cutoff_width,
        max_num_neighbors=model.max_num_neighbors,
    )


def test_adaptive_cutoff_selected_atoms(
    monkeypatch, tmp_path, adaptive_cutoff_test_systems
):
    """Tests the adaptive cutoff functionality."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )

    model_hypers = MODEL_HYPERS.copy()
    model_hypers["cutoff"] = 7.0
    model_hypers["max_num_neighbors"] = 2
    selected_atoms = Labels(
        names=["system", "atom"],
        values=torch.tensor([[0, 0]]),
    )
    model = PET(model_hypers, dataset_info)
    options = model.requested_neighbor_lists()[0]
    systems = [
        get_system_with_neighbor_lists(system, [options]).to(torch.float32)
        for system in adaptive_cutoff_test_systems
    ]

    systems_to_batch(
        systems,
        options,
        model.atomic_types,
        model.species_to_species_index,
        model.cutoff_width,
        max_num_neighbors=model.max_num_neighbors,
        selected_atoms=selected_atoms,
    )
