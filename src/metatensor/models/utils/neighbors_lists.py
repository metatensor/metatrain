from typing import List
from .data.system_to_ase import system_to_ase

import ase
import torch
from metatensor.torch import Labels, TensorBlock
from metatensor.torch.atomistic import (
    NeighborsListOptions,
    System,
    register_autograd_neighbors,
)


def get_system_with_neighbors_lists(
    system: System, neighbor_lists: List[NeighborsListOptions]
) -> System:
    """Attaches neighbor lists to a `System` object.

    :param system: The system for which to calculate neighbor lists.
    :param neighbor_lists: A list of `NeighborsListOptions` objects,
        each of which specifies the parameters for a neighbor list.

    :return: The `System` object with the neighbor lists added.
    """
    # Convert the system to an ASE atoms object
    atoms = system_to_ase(system)

    # Compute the neighbor lists
    for options in neighbor_lists:
        if options not in system.known_neighbors_lists():
            neighbor_list = _compute_single_neighbor_list(atoms, options).to(
                device=system.device, dtype=system.dtype
            )
            register_autograd_neighbors(system, neighbor_list)
            system.add_neighbors_list(options, neighbor_list)

    return system


def _compute_single_neighbor_list(
    atoms: ase.Atoms, options: NeighborsListOptions
) -> TensorBlock:
    # Computes a single neighbor list for an ASE atoms object

    nl = ase.neighborlist.NeighborList(
        cutoffs=[options.engine_cutoff] * len(atoms),
        skin=0.0,
        sorted=False,
        self_interaction=False,
        bothways=options.full_list,
        primitive=ase.neighborlist.NewPrimitiveNeighborList,
    )
    nl.update(atoms)

    cell = torch.from_numpy(atoms.cell[:])
    positions = torch.from_numpy(atoms.positions)

    samples = []
    distances = []
    cutoff2 = options.engine_cutoff * options.engine_cutoff
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        for j, offset in zip(indices, offsets):
            distance = positions[j] - positions[i] + offset.dot(cell)

            distance2 = torch.dot(distance, distance).item()

            if distance2 > cutoff2:
                continue

            samples.append((i, j, offset[0], offset[1], offset[2]))
            distances.append(distance.to(dtype=torch.float64))

    if len(distances) == 0:
        stacked_distances = torch.zeros((0, 3), dtype=positions.dtype)
        samples = torch.zeros((0, 5), dtype=torch.int32)
    else:
        samples = torch.tensor(samples, dtype=torch.int32)
        stacked_distances = torch.vstack(distances)

    return TensorBlock(
        values=stacked_distances.reshape(-1, 3, 1),
        samples=Labels(
            names=[
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            values=samples,
        ),
        components=[Labels.range("xyz", 3)],
        properties=Labels.range("distance", 1),
    )
