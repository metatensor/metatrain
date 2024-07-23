from typing import List

import torch
from metatensor.torch.atomistic import System


def concatenate_structures(systems: List[System]):

    positions = []
    centers = []
    neighbors = []
    species = []
    segment_indices = []
    edge_vectors = []
    cell_shifts = []
    node_counter = 0

    for i, system in enumerate(systems):
        positions.append(system.positions)
        species.append(system.types)
        segment_indices.append(
            torch.full((len(system.positions),), i, device=system.device)
        )

        assert len(system.known_neighbor_lists()) == 1
        neighbor_list = system.get_neighbor_list(system.known_neighbor_lists()[0])
        nl_values = neighbor_list.samples.values
        edge_vectors_system = neighbor_list.values.reshape(
            neighbor_list.values.shape[0], 3
        )

        centers.append(nl_values[:, 0] + node_counter)
        neighbors.append(nl_values[:, 1] + node_counter)
        edge_vectors.append(edge_vectors_system)

        cell_shifts.append(nl_values[:, 2:])

        node_counter += len(system.positions)

    positions = torch.cat(positions)
    centers = torch.cat(centers)
    neighbors = torch.cat(neighbors)
    species = torch.cat(species)
    segment_indices = torch.cat(segment_indices)
    edge_vectors = torch.cat(edge_vectors)
    cell_shifts = torch.cat(cell_shifts)

    return (
        positions,
        centers,
        neighbors,
        species,
        segment_indices,
        edge_vectors,
        cell_shifts,
    )
