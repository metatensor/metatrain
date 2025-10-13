from typing import List, Optional

import torch
from metatensor.torch import Labels
from metatomic.torch import NeighborListOptions, System

from metatrain.pet.modules.nef import (
    compute_reversed_neighbor_list,
    edge_array_to_nef,
    get_corresponding_edges,
    get_nef_indices,
)


def concatenate_structures(
    systems: list[System],
    neighbor_list_options: NeighborListOptions,
    selected_atoms: Optional[Labels] = None,
):
    positions: list[torch.Tensor] = []
    momenta: list[torch.Tensor] = []
    centers: list[torch.Tensor] = []
    masses: list[torch.Tensor] = []
    neighbors: list[torch.Tensor] = []
    species: list[torch.Tensor] = []
    cell_shifts: list[torch.Tensor] = []
    cells: list[torch.Tensor] = []
    system_indices: list[torch.Tensor] = []
    atom_indices: list[torch.Tensor] = []
    node_counter = 0

    for i, system in enumerate(systems):
        assert len(system.known_neighbor_lists()) >= 1, "no neighbor list found"
        neighbor_list = system.get_neighbor_list(neighbor_list_options)
        nl_values = neighbor_list.samples.values

        centers_values = nl_values[:, 0]
        neighbors_values = nl_values[:, 1]
        cell_shifts_values = nl_values[:, 2:]

        if selected_atoms is not None:
            system_selected_atoms = selected_atoms.values[:, 1][
                selected_atoms.values[:, 0] == i
            ]
            unique_centers = torch.unique(centers_values)
            system_selected_atoms = torch.unique(
                torch.cat([system_selected_atoms, unique_centers])
            )
            # calculate the mapping from the ghost atoms to the real atoms
            ghost_to_real_index = torch.full(
                [
                    int(unique_centers.max()) + 1,
                ],
                -1,
                device=centers_values.device,
                dtype=centers_values.dtype,
            )
            for j, unique_center_index in enumerate(unique_centers):
                ghost_to_real_index[unique_center_index] = j

            centers_values = ghost_to_real_index[centers_values]
            neighbors_values = ghost_to_real_index[neighbors_values]
        else:
            system_selected_atoms = torch.arange(
                len(system), device=system.positions.device
            )

        positions.append(system.positions[system_selected_atoms])
        if "momenta" not in system.known_data():
            raise ValueError(
                "System does not contain momenta data, which is required for FlashMD."
            )
        tmap = system.get_data("momenta")
        block = tmap[0]
        momenta.append(block.values[system_selected_atoms].squeeze(-1))
        species.append(system.types[system_selected_atoms])

        centers.append(centers_values + node_counter)
        tmap = system.get_data("masses")
        block = tmap[0]
        masses.append(block.values[system_selected_atoms].squeeze(-1))
        neighbors.append(neighbors_values + node_counter)
        cell_shifts.append(cell_shifts_values)

        cells.append(system.cell)

        node_counter += len(system_selected_atoms)
        system_indices.append(
            torch.full((len(system_selected_atoms),), i, device=system.positions.device)
        )
        atom_indices.append(
            torch.arange(len(system_selected_atoms), device=system.positions.device)
        )

    positions = torch.cat(positions)
    momenta = torch.cat(momenta)
    centers = torch.cat(centers)
    masses = torch.cat(masses)
    neighbors = torch.cat(neighbors)
    species = torch.cat(species)
    cells = torch.stack(cells)
    cell_shifts = torch.cat(cell_shifts)
    system_indices = torch.cat(system_indices)
    atom_indices = torch.cat(atom_indices)

    sample_values = torch.stack(
        [system_indices, atom_indices],
        dim=1,
    )
    sample_labels = Labels(
        names=["system", "atom"],
        values=sample_values,
    )

    return (
        positions,
        momenta,
        centers,
        masses,
        neighbors,
        species,
        cells,
        cell_shifts,
        system_indices,
        sample_labels,
    )


def systems_to_batch(
    systems: List[System],
    options: NeighborListOptions,
    all_species_list: List[int],
    species_to_species_index: torch.Tensor,
    selected_atoms: Optional[Labels] = None,
):
    """
    Converts a list of systems to a batch required for the PET model.
    The batch consists of the following tensors:
    - `element_indices_nodes`: The atomic species of the central atoms
    - `momenta`: The momenta of the central atoms.
    - `element_indices_neighbors`: The atomic species of the neighboring atoms
    - `edge_vectors`: The cartedian edge vectors between the central atoms and their
      neighbors
    - `padding_mask`: A padding mask indicating which neighbors are real, and which are
      padded
    - `neighbors_index`: The indices of the neighboring atoms for each central atom
    - `num_neghbors`: The number of neighbors for each central atom
    - `reversed_neighbor_list`: The reversed neighbor list for each central atom
    """
    # save_system(systems[0], options, selected_atoms)
    (
        positions,
        momenta,
        centers,
        masses,
        neighbors,
        species,
        cells,
        cell_shifts,
        system_indices,
        sample_labels,
    ) = concatenate_structures(systems, options, selected_atoms)

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

    if selected_atoms is not None:
        num_nodes = int(centers.max()) + 1
    else:
        num_nodes = len(positions)

    # Convert to NEF (Node-Edge-Feature) format:
    nef_indices, nef_to_edges_neighbor, nef_mask = get_nef_indices(
        centers, num_nodes, max_edges_per_node
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
        positions,
        momenta,
        masses,
        element_indices_neighbors,
        edge_vectors,
        nef_mask,
        neighbors_index,
        reversed_neighbor_list,
        system_indices,
        sample_labels,
    )
