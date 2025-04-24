from typing import List, Optional, Tuple

import torch
from metatensor.torch import Labels, TensorBlock
from metatensor.torch.atomistic import NeighborListOptions, System

from .nef import (
    compute_reversed_neighbor_list,
    edge_array_to_nef,
    get_corresponding_edges,
    get_nef_indices,
)


def concatenate_structures(
    systems: List[System],
    neighbor_list_options: NeighborListOptions,
):
    positions: List[torch.Tensor] = []
    centers: List[torch.Tensor] = []
    neighbors: List[torch.Tensor] = []
    species: List[torch.Tensor] = []
    cell_shifts: List[torch.Tensor] = []
    cells: List[torch.Tensor] = []
    node_counter = 0

    for system in systems:
        assert len(system.known_neighbor_lists()) >= 1, "no neighbor list found"
        neighbor_list = system.get_neighbor_list(neighbor_list_options)
        nl_values = neighbor_list.samples.values

        centers_values = nl_values[:, 0]
        neighbors_values = nl_values[:, 1]
        cell_shifts_values = nl_values[:, 2:]

        positions.append(system.positions)
        species.append(system.types)

        centers.append(centers_values + node_counter)
        neighbors.append(neighbors_values + node_counter)
        cell_shifts.append(cell_shifts_values)

        cells.append(system.cell)

        node_counter += len(system.positions)

    positions = torch.cat(positions)
    centers = torch.cat(centers)
    neighbors = torch.cat(neighbors)
    species = torch.cat(species)
    cells = torch.stack(cells)
    cell_shifts = torch.cat(cell_shifts)

    return (
        positions,
        centers,
        neighbors,
        species,
        cells,
        cell_shifts,
    )


def remap_neighborlists(
    systems: List[System],
    neighbor_list_options: NeighborListOptions,
    selected_atoms: Optional[Labels] = None,
) -> List[System]:
    """
    This function remaps the neighbor lists from the LAMMPS format
    to ASE format. The main difference between LAMMPS and ASE neighbor
    lists is that LAMMPS treats both real and ghost atoms as real atoms.
    Because of that, there is a certain degree of duplication in the data.
    Moreover, in the case of domain decomposition, the indices of the atoms
    may not be contiguous.This function removes the ghost atoms from the
    positions and types, while remapping the indices of the neighbor lists
    to a contiguous format.
    """

    new_systems: List[System] = []
    for i, system in enumerate(systems):
        assert len(system.known_neighbor_lists()) >= 1, (
            "the system must have at least one neighbor list"
        )
        if selected_atoms is not None:
            selected_atoms_index = selected_atoms.values[:, 1][
                selected_atoms.values[:, 0] == i
            ]
        else:
            selected_atoms_index = torch.arange(
                len(system), device=system.positions.device
            )
        nl = system.get_neighbor_list(neighbor_list_options)
        nl_values = nl.samples.values

        centers = nl_values[:, 0]
        neighbors = nl_values[:, 1]
        cell_shifts = nl_values[:, 2:]

        unique_neighbors_index = torch.unique(centers)
        unique_index = torch.unique(
            torch.cat((selected_atoms_index, unique_neighbors_index))
        )

        centers, neighbors, unique_neighbors_index = remap_to_contiguous_indexing(
            centers,
            neighbors,
            unique_neighbors_index,
            unique_index,
            device=system.positions.device,
        )

        index = torch.argsort(centers, stable=True)

        centers = centers[index].contiguous()
        neighbors = neighbors[index].contiguous()
        cell_shifts = cell_shifts[index].contiguous()
        positions = system.positions[unique_index]
        types = system.types[unique_index]
        distances = nl.values[index].contiguous()

        new_system = System(
            positions=positions,
            types=types,
            cell=system.cell,
            pbc=system.pbc,
        )

        new_nl = TensorBlock(
            samples=Labels(
                names=nl.samples.names,
                values=torch.cat(
                    (
                        centers.unsqueeze(1),
                        neighbors.unsqueeze(1),
                        cell_shifts,
                    ),
                    dim=1,
                ),
            ),
            components=nl.components,
            properties=nl.properties,
            values=distances,
        )
        new_system.add_neighbor_list(neighbor_list_options, new_nl)
        new_systems.append(new_system)

    return new_systems


def remap_to_contiguous_indexing(
    centers: torch.Tensor,
    neighbors: torch.Tensor,
    unique_neighbors_index: torch.Tensor,
    unique_index: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    This helper function remaps the indices of center and neighbor atoms
    from arbitrary indexing to contgious indexing, i.e.

    from
    0, 1, 2, 54, 55, 56
    to
    0, 1, 2, 3, 4, 5.

    This remapping is required by internal implementation of PET neighbor lists, where
    indices of the atoms cannot exceed the total amount of atoms in the system.

    Shifted indices come from LAMMPS neighborlists in the case of domain decomposition
    enabled, since they contain not only the atoms in the unit cell, but also so-called
    ghost atoms, which may have a different indexing. Thus, to avoid further errors, we
    remap the indices to a contiguous format.

    """
    index_map = torch.empty(
        int(unique_index.max().item()) + 1, dtype=torch.int64, device=device
    )
    index_map[unique_index] = torch.arange(len(unique_index), device=device)
    centers = index_map[centers]
    neighbors = index_map[neighbors]
    unique_neighbors_index = index_map[unique_neighbors_index]
    return centers, neighbors, unique_neighbors_index


def systems_to_batch(
    systems: List[System],
    options: NeighborListOptions,
    all_species_list: List[int],
    system_indices: torch.Tensor,
    species_to_species_index: torch.Tensor,
    selected_atoms: Optional[Labels] = None,
):
    """
    Converts a list of systems to a batch required for the NativePET model.
    The batch consists of the following tensors:
    - `element_indices_nodes`: The atomic species of the central atoms
    - `element_indices_neighbors`: The atomic species of the neighboring atoms
    - `edge_vectors`: The cartedian edge vectors between the central atoms and their
      neighbors
    - `padding_mask`: A padding mask indicating which neighbors are real, and which are
      padded
    - `neighbors_index`: The indices of the neighboring atoms for each central atom
    - `num_neghbors`: The number of neighbors for each central atom
    - `reversed_neighbor_list`: The reversed neighbor list for each central atom
    """
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
    num_neghbors = torch.bincount(centers)
    if num_neghbors.numel() == 0:  # no edges
        max_edges_per_node = 0
    else:
        max_edges_per_node = int(torch.max(num_neghbors))

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

    reversed_neighbor_list = compute_reversed_neighbor_list(
        nef_indices, corresponding_edges, nef_mask
    )
    neighbors_index = edge_array_to_nef(neighbors, nef_indices).to(torch.int64)
    return (
        element_indices_nodes,
        element_indices_neighbors,
        edge_vectors,
        nef_mask,
        neighbors_index,
        num_neghbors,
        reversed_neighbor_list,
    )
