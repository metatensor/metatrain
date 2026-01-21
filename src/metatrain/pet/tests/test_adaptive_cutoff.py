import random

import numpy as np
import pytest
import torch
from ase.build import bulk
from metatomic.torch import ModelOutput, NeighborListOptions, System, systems_to_torch

from metatrain.pet import PET
from metatrain.pet.modules.adaptive_cutoff import (
    get_adaptive_cutoffs,
    get_effective_num_neighbors,
    get_gaussian_cutoff_weights,
)
from metatrain.pet.modules.structures import concatenate_structures
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import (
    get_energy_target_info,
)
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import MODEL_HYPERS


@pytest.mark.parametrize("num_neighbors_adaptive", [8, 16, 32, 64, None])
def test_adaptive_cutoff_functionality(num_neighbors_adaptive):
    """Tests that adaptive cutoff model evaluation runs without errors."""
    torch.manual_seed(0)
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )
    hypers = MODEL_HYPERS.copy()
    hypers["num_neighbors_adaptive"] = num_neighbors_adaptive
    hypers["cutoff"] = 10.0

    model = PET(hypers, dataset_info)

    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]]),
        cell=torch.eye(3),
        pbc=torch.tensor([True, True, True]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    outputs = {"energy": ModelOutput(per_atom=False)}
    _ = model([system], outputs)


def test_effective_num_neighbors():
    """Tests that the effective number of neighbors calculation is correct."""
    edge_distances = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    centers = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    probe_cutoffs = torch.tensor([1.0, 2.0, 3.0, 4.0])
    num_nodes = 3

    effective_num_neighbors = get_effective_num_neighbors(
        edge_distances,
        probe_cutoffs,
        centers,
        num_nodes,
    )

    effective_num_neighbors_expected = torch.tensor(
        [
            [0.0000, 1.5000, 2.0000, 2.0000],
            [0.0000, 0.0000, 1.5000, 3.0000],
            [0.0000, 0.0000, 0.0000, 0.0000],
        ]
    )
    assert torch.allclose(effective_num_neighbors, effective_num_neighbors_expected)


def test_gaussian_cutoff_weights():
    """Tests that the Gaussian cutoff weights calculation is correct."""
    effective_num_neighbors = torch.tensor(
        [
            [0.0000, 1.5000, 2.0000, 2.0000],
            [0.0000, 0.0000, 1.5000, 3.0000],
            [0.0000, 0.0000, 0.0000, 0.0000],
        ]
    )
    num_neighbors_adaptive = 2.0

    cutoff_weights = get_gaussian_cutoff_weights(
        effective_num_neighbors,
        num_neighbors_adaptive,
    )

    cutoff_weights_expected = torch.tensor(
        [
            [
                1.686352491379e-01,
                3.581513762474e-01,
                3.354909420013e-01,
                1.377224177122e-01,
            ],
            [
                0.000000000000e00,
                1.038051098585e-01,
                5.644838809967e-01,
                3.317110538483e-01,
            ],
            [
                0.000000000000e00,
                4.980000856136e-10,
                2.557745575905e-01,
                7.442253828049e-01,
            ],
        ]
    )
    assert torch.allclose(cutoff_weights, cutoff_weights_expected)


@pytest.mark.parametrize("num_neighbors_adaptive", [8, 16, 24, 32, 48])
def test_adapted_cutoffs(num_neighbors_adaptive):
    """Tests that adaptive cutoff model evaluation runs without errors
    and produces reasonable cutoffs that approximately ensure the desired
    number of neighbors (within some tolerance of +/- 10 neighbors)."""
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    options = NeighborListOptions(cutoff=10.0, full_list=True, strict=True)

    atoms = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)
    atoms.rattle(0.1)
    system = systems_to_torch(atoms)

    system = get_system_with_neighbor_lists(system, [options])
    systems = [system]

    (
        positions,
        centers,
        neighbors,
        species,
        cells,
        cell_shifts,
        system_indices,
        sample_labels,
    ) = concatenate_structures(systems, options, None)

    # somehow the backward of this operation is very slow at evaluation,
    # where there is only one cell, therefore we simplify the calculation
    # for that case
    if len(cells) == 1:
        cell_contributions = cell_shifts.to(cells.dtype) @ cells[0]
    else:
        cell_contributions = torch.einsum(
            "ab, abc -> ac",
            cell_shifts.to(cells.dtype),
            cells[system_indices[centers]],
        )
    edge_vectors = positions[neighbors] - positions[centers] + cell_contributions
    edge_distances = torch.norm(edge_vectors, dim=-1) + 1e-15
    num_nodes = len(positions)

    atomic_cutoffs = get_adaptive_cutoffs(
        centers,
        edge_distances,
        num_neighbors_adaptive,
        num_nodes,
        options.cutoff,
        cutoff_width=0.5,
    )
    pair_cutoffs = (atomic_cutoffs[centers] + atomic_cutoffs[neighbors]) / 2.0
    cutoff_mask = edge_distances <= pair_cutoffs
    pair_cutoffs = pair_cutoffs[cutoff_mask]
    centers = centers[cutoff_mask]
    adapted_num_neighbors = torch.bincount(centers, minlength=num_nodes)
    diff = torch.abs(adapted_num_neighbors - num_neighbors_adaptive)
    assert torch.all(diff <= 10)
