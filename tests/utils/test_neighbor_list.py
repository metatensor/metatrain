from pathlib import Path

import ase.build
import ase.neighborlist
import numpy as np
import pytest
import torch
from metatomic.torch import NeighborListOptions, System

from metatrain.utils.data.readers.ase import read_systems
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


RESOURCES_PATH = Path(__file__).parents[1] / "resources"


def _system(atoms, device="cpu", requires_grad=False):
    cell = np.array(atoms.cell[:])
    # non-periodic directions are represented by zero cell vectors
    cell[~atoms.pbc] = 0.0
    return System(
        types=torch.tensor(atoms.numbers, dtype=torch.int32, device=device),
        positions=torch.tensor(
            atoms.positions, dtype=torch.float64, device=device
        ).requires_grad_(requires_grad),
        cell=torch.tensor(cell, dtype=torch.float64, device=device),
        pbc=torch.tensor(atoms.pbc, device=device),
    )


def _pair_set(samples):
    # makes (i, j, S) and (j, i, -S) the same pair, to compare full and half lists
    return {
        min((i, j, a, b, c), (j, i, -a, -b, -c)) for i, j, a, b, c in samples.tolist()
    }


def _structures():
    slab = ase.build.fcc111("Pt", size=(2, 2, 3), vacuum=6.0)

    wire = ase.build.bulk("Cu", "fcc", a=3.6, cubic=True).repeat((1, 1, 3))
    wire.pbc = [False, False, True]
    wire.center(vacuum=6.0, axis=(0, 1))

    return {
        "molecule": ase.build.molecule("C6H6"),
        # small cell, atoms are neighbors of their own periodic images
        "bulk": ase.build.bulk("Si", "diamond", a=5.43),
        "slab": slab,
        "wire": wire,
    }


@pytest.mark.parametrize("name", ["molecule", "bulk", "slab", "wire"])
@pytest.mark.parametrize("full_list", [True, False])
def test_neighbor_list_matches_ase(name, full_list):
    """The neighbor list matches ASE, including for partial periodic boundaries."""
    atoms = _structures()[name]
    cutoff = 5.0

    options = NeighborListOptions(cutoff=cutoff, full_list=full_list, strict=True)
    system = get_system_with_neighbor_lists(_system(atoms), [options])
    neighbors = system.get_neighbor_list(options)

    i, j, S = ase.neighborlist.neighbor_list("ijS", atoms, cutoff=cutoff)
    reference = np.concatenate([i[:, None], j[:, None], S], axis=1)

    if full_list:
        assert len(neighbors.samples) == len(reference)
    else:
        # each pair is only present once in a half list
        assert 2 * len(neighbors.samples) == len(reference)
    assert _pair_set(neighbors.samples.values) == _pair_set(torch.tensor(reference))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_neighbor_list_device(device):
    """The neighbor list is computed on the device of the system."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    atoms = _structures()["slab"]
    options = NeighborListOptions(cutoff=5.0, full_list=False, strict=True)

    system = get_system_with_neighbor_lists(_system(atoms, device=device), [options])
    neighbors = system.get_neighbor_list(options)
    assert neighbors.values.device.type == device
    assert neighbors.samples.values.device.type == device

    reference = get_system_with_neighbor_lists(_system(atoms), [options])
    reference = reference.get_neighbor_list(options)
    assert _pair_set(neighbors.samples.values.cpu()) == _pair_set(
        reference.samples.values
    )


def test_neighbor_list_gradients():
    """Gradients flow from the neighbor list distances back to the positions."""
    atoms = ase.build.bulk("Si", "diamond", a=5.43).repeat((2, 2, 2))
    atoms.rattle(0.05, seed=0)
    options = NeighborListOptions(cutoff=4.0, full_list=True, strict=True)

    system = get_system_with_neighbor_lists(
        _system(atoms, requires_grad=True), [options]
    )
    distances = system.get_neighbor_list(options).values
    distances.pow(2).sum().backward()

    assert system.positions.grad is not None
    assert torch.any(system.positions.grad != 0)


def test_attach_neighbor_lists():
    filename = RESOURCES_PATH / "qm9_reduced_100.xyz"
    systems = read_systems(filename)

    requested_neighbor_lists = [
        NeighborListOptions(cutoff=4.0, full_list=True, strict=True),
        NeighborListOptions(cutoff=5.0, full_list=False, strict=True),
        NeighborListOptions(cutoff=6.0, full_list=True, strict=True),
    ]

    new_system = get_system_with_neighbor_lists(systems[0], requested_neighbor_lists)

    assert requested_neighbor_lists[0] in new_system.known_neighbor_lists()
    assert requested_neighbor_lists[1] in new_system.known_neighbor_lists()
    assert requested_neighbor_lists[2] in new_system.known_neighbor_lists()

    extraneous_nl = NeighborListOptions(cutoff=5.0, full_list=True, strict=True)
    assert extraneous_nl not in new_system.known_neighbor_lists()

    for nl_options in new_system.known_neighbor_lists():
        nl = new_system.get_neighbor_list(nl_options)
        assert nl.samples.names == [
            "first_atom",
            "second_atom",
            "cell_shift_a",
            "cell_shift_b",
            "cell_shift_c",
        ]
        assert len(nl.values.shape) == 3
