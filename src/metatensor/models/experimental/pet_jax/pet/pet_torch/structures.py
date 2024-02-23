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
    node_counter = 0

    for i, system in enumerate(systems):
        positions.append(system.positions)
        species.append(system.species)
        segment_indices.append(torch.full((len(system.positions),), i))

        assert len(system.known_neighbors_lists()) == 1
        neighbor_list = system.get_neighbors_list(system.known_neighbors_lists()[0])
        nl_values = neighbor_list.samples.values
        edge_vectors_system = neighbor_list.values.reshape(-1, 3)

        centers.append(nl_values[:, 0] + node_counter)
        neighbors.append(nl_values[:, 1] + node_counter)
        edge_vectors.append(edge_vectors_system)

        node_counter += len(system.positions)

    positions = torch.cat(positions)
    centers = torch.cat(centers)
    neighbors = torch.cat(neighbors)
    species = torch.cat(species)
    segment_indices = torch.cat(segment_indices)
    edge_vectors = torch.cat(edge_vectors)

    return positions, centers, neighbors, species, segment_indices, edge_vectors
