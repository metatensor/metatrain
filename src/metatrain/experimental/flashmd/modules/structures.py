from typing import List

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
):
    """
    Concatenate a list of systems into a single batch.

    :param systems: List of systems to concatenate.
    :param neighbor_list_options: Options for the neighbor list.
    :return: A tuple containing the concatenated positions, momenta, centers, neighbors,
        species, cells, cell shifts, system indices, and sample labels.
    """
    positions: list[torch.Tensor] = []
    momenta: list[torch.Tensor] = []
    centers: list[torch.Tensor] = []
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

        system_size = len(system)
        positions.append(system.positions)
        species.append(system.types)

        if "momenta" not in system.known_data():
            raise ValueError(
                "System does not contain momenta data, which is required for FlashMD."
            )
        tmap = system.get_data("momenta")
        block = tmap[0]
        momenta.append(block.values.squeeze(-1))

        centers.append(centers_values + node_counter)
        neighbors.append(neighbors_values + node_counter)
        cell_shifts.append(cell_shifts_values)

        cells.append(system.cell)

        node_counter += system_size
        system_indices.append(
            torch.full((system_size,), i, device=system.positions.device)
        )
        atom_indices.append(torch.arange(system_size, device=system.positions.device))

    positions = torch.cat(positions)
    momenta = torch.cat(momenta)
    centers = torch.cat(centers)
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

    num_nodes = len(positions)
    num_neighbors = torch.bincount(centers, minlength=num_nodes)

    # this logic shouldn't be needed thanks to `minlength` above, but just to be safe:
    max_edges_per_node = (
        int(torch.max(num_neighbors)) if num_neighbors.numel() > 0 else 0
    )

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

    # These are the two arrays we need for message passing with edge reversals,
    # if indexing happens in a two-dimensional way:
    # edges_ji = edges_ij[reversed_neighbor_list, neighbors_index]
    reversed_neighbor_list = compute_reversed_neighbor_list(
        nef_indices, corresponding_edges, nef_mask
    )
    neighbors_index = edge_array_to_nef(neighbors, nef_indices).to(torch.int64)

    # Here, we compute the array that allows indexing into a flattened
    # version of the edge array (where the first two dimensions are merged):
    reverse_neighbor_index = (
        neighbors_index * neighbors_index.shape[1] + reversed_neighbor_list
    )
    # At this point, we have `reverse_neighbor_index[~nef_mask] = 0`, which however
    # creates too many of the same index which slows down backward enormously.
    # (See see https://github.com/pytorch/pytorch/issues/41162)
    # We therefore replace the padded indices with a sequence of unique indices.
    reverse_neighbor_index[~nef_mask] = torch.arange(
        int(torch.sum(~nef_mask)), device=reverse_neighbor_index.device
    )

    return (
        element_indices_nodes,
        momenta,
        element_indices_neighbors,
        edge_vectors,
        nef_mask,
        reverse_neighbor_index,
        system_indices,
        sample_labels,
    )
