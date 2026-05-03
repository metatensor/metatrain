import random

import numpy as np
import pytest
import torch
from ase.build import bulk
from metatomic.torch import ModelOutput, NeighborListOptions, System, systems_to_torch

from metatrain.pet import PET
from metatrain.pet.modules.adaptive_cutoff import (
    get_adaptive_cutoffs_grid,
    get_adaptive_cutoffs_solver,
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


@pytest.mark.parametrize("adaptive_cutoff_method", ["solver", "grid"])
@pytest.mark.parametrize("num_neighbors_adaptive", [8, 16, 32, 64, None])
def test_adaptive_cutoff_functionality(num_neighbors_adaptive, adaptive_cutoff_method):
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
    hypers["adaptive_cutoff_method"] = adaptive_cutoff_method
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


@pytest.mark.parametrize(
    "get_adaptive_cutoffs",
    [get_adaptive_cutoffs_solver, get_adaptive_cutoffs_grid],
    ids=["solver", "grid"],
)
@pytest.mark.parametrize("num_neighbors_adaptive", [8, 16, 24, 32, 48])
def test_adapted_cutoffs(num_neighbors_adaptive, get_adaptive_cutoffs):
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
    ) = concatenate_structures(systems, options)

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


@pytest.mark.parametrize(
    "get_adaptive_cutoffs",
    [get_adaptive_cutoffs_solver, get_adaptive_cutoffs_grid],
    ids=["solver", "grid"],
)
def test_adaptive_cutoff_gradients(get_adaptive_cutoffs):
    """``d r_bar / d edge_distances`` is correct.

    The solver bypasses the Newton iterations on the backward path via an
    implicit-function-theorem step; this test verifies the resulting gradient
    matches finite differences. The grid path's gradient (through the Gaussian
    weighting) is checked too.
    """
    torch.manual_seed(0)

    # Small handcrafted configuration so r_bar sits comfortably inside the
    # solver's [max_cutoff/16, max_cutoff] clamp band: 4 neighbors with
    # nbar=4 puts r_bar near the median neighbor distance, well away from
    # the clamps.
    centers = torch.tensor([0, 0, 0, 0, 1, 1])
    edge_distances = torch.tensor(
        [1.0, 1.7, 2.3, 2.9, 1.4, 2.1], dtype=torch.float64, requires_grad=True
    )
    num_nodes = 2
    max_cutoff = 5.0
    cutoff_width = 0.5
    nbar = 3.0

    def fn(d):
        return get_adaptive_cutoffs(
            centers, d, nbar, num_nodes, max_cutoff, cutoff_width=cutoff_width
        )

    # Sanity-check that r_bar is interior to the clamp band so the test is
    # actually exercising the IFT branch (and not the .clamp boundaries).
    with torch.no_grad():
        r_bar = fn(edge_distances)
        assert torch.all(r_bar > max_cutoff / 16.0 + 1e-3)
        assert torch.all(r_bar < max_cutoff - 1e-3)

    # gradcheck calls fn many times with perturbed inputs; eps is chosen to
    # match the typical scale of the bump active band.
    assert torch.autograd.gradcheck(
        fn, (edge_distances,), eps=1e-5, atol=1e-4, rtol=1e-3, fast_mode=True
    )


@pytest.mark.parametrize("adaptive_cutoff_method", ["solver", "grid"])
def test_adaptive_cutoff_empty_system(adaptive_cutoff_method):
    """Tests that the model can handle an empty system."""

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
    hypers["num_neighbors_adaptive"] = 8
    hypers["adaptive_cutoff_method"] = adaptive_cutoff_method
    hypers["cutoff"] = 10.0

    model = PET(hypers, dataset_info)

    system = System(
        types=torch.tensor([], dtype=torch.long),
        positions=torch.empty((0, 3)),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    outputs = {"energy": ModelOutput(per_atom=False)}
    energy = model([system], outputs)["energy"].block().values.squeeze(-1)
    assert torch.numel(energy) == 0


@pytest.mark.parametrize("adaptive_cutoff_method", ["solver", "grid"])
def test_adaptive_cutoff_isolated_atom(adaptive_cutoff_method):
    """Tests that the model can predict energies for an isolated atom
    with adaptive cutoff enabled."""

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
    hypers["num_neighbors_adaptive"] = 8
    hypers["adaptive_cutoff_method"] = adaptive_cutoff_method
    hypers["cutoff"] = 10.0

    model = PET(hypers, dataset_info)

    system = System(
        types=torch.tensor([6]),
        positions=torch.tensor([[0.0, 0.0, 0.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    outputs = {"energy": ModelOutput(per_atom=False)}
    _ = model([system], outputs)


@pytest.mark.parametrize("adaptive_cutoff_method", ["solver", "grid"])
@pytest.mark.parametrize("cutoff", [10.0, 5.0])
def test_adaptive_cutoff_dissociated_atoms(cutoff, adaptive_cutoff_method):
    """Tests that the model can predict energies for an isolated atom
    with adaptive cutoff enabled."""

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
    hypers["num_neighbors_adaptive"] = 8
    hypers["adaptive_cutoff_method"] = adaptive_cutoff_method
    hypers["cutoff"] = cutoff

    model = PET(hypers, dataset_info)

    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 7.5]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    outputs = {"energy": ModelOutput(per_atom=False)}
    _ = model([system], outputs)
