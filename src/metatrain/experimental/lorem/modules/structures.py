from typing import List, Tuple

import torch
from metatomic.torch import NeighborListOptions, System


def concatenate_structures(
    systems: List[System], neighbor_list_options: NeighborListOptions
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Concatenate a list of systems and their neighbor lists into one batch.

    :param systems: Systems to concatenate.
    :param neighbor_list_options: Neighbor-list options used when the lists
        were attached to each system.
    :return: Concatenated positions, center indices, neighbor indices, species,
        cells, and cell shifts.
    """
    positions: List[torch.Tensor] = []
    centers: List[torch.Tensor] = []
    neighbors: List[torch.Tensor] = []
    species: List[torch.Tensor] = []
    cell_shifts: List[torch.Tensor] = []
    cells: List[torch.Tensor] = []
    node_counter = 0

    for system in systems:
        positions.append(system.positions)
        species.append(system.types)

        neighbor_list = system.get_neighbor_list(neighbor_list_options)
        nl_values = neighbor_list.samples.values

        centers.append(nl_values[:, 0] + node_counter)
        neighbors.append(nl_values[:, 1] + node_counter)
        cell_shifts.append(nl_values[:, 2:])
        cells.append(system.cell)

        node_counter += len(system.positions)

    return (
        torch.cat(positions),
        torch.cat(centers),
        torch.cat(neighbors),
        torch.cat(species),
        torch.stack(cells),
        torch.cat(cell_shifts),
    )
