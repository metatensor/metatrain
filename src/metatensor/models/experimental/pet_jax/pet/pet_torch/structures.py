from typing import List

import torch
from metatensor.torch.atomistic import System


def concatenate_structures(systems: List[System]):

    positions = []
    centers = []
    neighbors = []
    cell_shifts = []
    species = []
    cells = []
    segment_indices = []
    node_counter = 0

    for i, system in enumerate(systems):
        positions.append(system.positions)
        species.append(system.species)
        cells.append(system.cell)
        segment_indices.append(torch.full((len(system.positions),), i))

        assert len(system.known_neighbors_lists() == 1)
        neighbor_list = system.get_neighbors_lists()[0]
        nl_values = neighbor_list.block().values

        centers.append(nl_values[0] + node_counter)
        neighbors.append(nl_values[1] + node_counter)
        cell_shifts.append(nl_values[2:])

        node_counter += len(system.positions)

    positions = torch.cat(positions)
    centers = torch.cat(centers)
    neighbors = torch.cat(neighbors)
    cell_shifts = torch.cat(cell_shifts)
    species = torch.cat(species)
    cells = torch.stack(cells)
    segment_indices = torch.cat(segment_indices)

    return positions, centers, neighbors, cell_shifts, species, cells, segment_indices
