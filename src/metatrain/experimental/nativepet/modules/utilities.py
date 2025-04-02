from typing import Dict, List

import torch
from metatensor.torch.atomistic import NeighborListOptions, System

from .nef import (
    compute_neighbors_pos,
    edge_array_to_nef,
    get_corresponding_edges,
    get_nef_indices,
)
from .structures import concatenate_structures


def cutoff_func(grid: torch.Tensor, r_cut: float, delta: float):
    mask_bigger = grid >= r_cut
    mask_smaller = grid <= r_cut - delta
    grid = (grid - r_cut + delta) / delta
    f = 0.5 + 0.5 * torch.cos(torch.pi * grid)

    f[mask_bigger] = 0.0
    f[mask_smaller] = 1.0
    return f


class NeverRun(torch.nn.Module):
    """Dummy torch module to make torchscript happy.
    This model should never be run"""

    def __init__(self):
        super(NeverRun, self).__init__()

    def forward(self, x) -> torch.Tensor:
        raise RuntimeError("This model should never be run")


def systems_to_batch_dict(
    systems: List[System],
    options: NeighborListOptions,
    all_species_list: List[int],
    species_to_species_index: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Converts a list of systems to a batch dictionary,
    required for the NativePET model. The batch dictionary
    consists of the following keys:
    - `central_species`: The atomic species of the central atoms
    - `neighbor_species`: The atomic species of the neighboring atoms
    - `x`: The cartedian edge vectors between the central atoms and their neighbors
    - `mask`: A padding mask indicating which neighbors are real, and which are padded
    - `neighbors_index`: The indices of the neighboring atoms for each central atom
    - `nums`: The number of neighbors for each central atom
    - `neighbors_pos`: The reversed neighbor list
    - `batch`: The batch indices for each atom
    """
    device = systems[0].positions.device
    system_indices = torch.concatenate(
        [
            torch.full(
                (len(system),),
                i_system,
                device=device,
            )
            for i_system, system in enumerate(systems)
        ],
    )
    (
        positions,
        centers,
        neighbors,
        species,
        cells,
        cell_shifts,
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
    bincount = torch.bincount(centers)
    if bincount.numel() == 0:  # no edges
        max_edges_per_node = 0
    else:
        max_edges_per_node = int(torch.max(bincount))

    # Convert to NEF (Node-Edge-Feature) format:
    nef_indices, nef_to_edges_neighbor, nef_mask = get_nef_indices(
        centers, len(positions), max_edges_per_node
    )

    # Element indices
    element_indices_nodes = species_to_species_index[species]
    element_indices_neighbors = element_indices_nodes[neighbors]

    # Send everything to NEF:
    edge_vectors = edge_array_to_nef(edge_vectors, nef_indices)
    element_indices_neighbors = edge_array_to_nef(
        element_indices_neighbors, nef_indices
    )

    corresponding_edges = get_corresponding_edges(
        torch.concatenate(
            [centers.unsqueeze(-1), neighbors.unsqueeze(-1), cell_shifts],
            dim=-1,
        )
    )

    neighbors_pos = compute_neighbors_pos(nef_indices, corresponding_edges, nef_mask)

    native_batch_dict = {
        "central_species": element_indices_nodes,
        "mask": torch.logical_not(nef_mask),
        "x": edge_vectors,
        "neighbor_species": element_indices_neighbors,
        "neighbors_index": edge_array_to_nef(neighbors, nef_indices).to(torch.int64),
        "nums": bincount,
        "neighbors_pos": neighbors_pos,
        "batch": system_indices,
    }
    return native_batch_dict
