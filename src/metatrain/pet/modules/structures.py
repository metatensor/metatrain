from typing import List, Optional, Tuple

import torch
from metatensor.torch import Labels
from metatomic.torch import NeighborListOptions, System

from .adaptive_cutoff import get_adaptive_cutoffs
from .nef import (
    compute_reversed_neighbor_list,
    edge_array_to_nef,
    get_corresponding_edges,
    get_nef_indices,
)
from .utilities import cutoff_func_bump, cutoff_func_cosine


def concatenate_structures(
    systems: List[System],
    neighbor_list_options: NeighborListOptions,
    selected_atoms: Optional[Labels] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Labels,
]:
    """
    Concatenate a list of systems into a single batch.

    :param systems: List of systems to concatenate.
    :param neighbor_list_options: Options for the neighbor list.
    :param selected_atoms: Optional labels of selected atoms to include in the batch.
    :return: A tuple containing the concatenated positions, centers, neighbors,
        species, cells, cell shifts, system indices, and sample labels.
    """

    positions: List[torch.Tensor] = []
    centers: List[torch.Tensor] = []
    neighbors: List[torch.Tensor] = []
    species: List[torch.Tensor] = []
    cell_shifts: List[torch.Tensor] = []
    cells: List[torch.Tensor] = []
    system_indices: List[torch.Tensor] = []
    atom_indices: List[torch.Tensor] = []
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
            if torch.numel(unique_centers) == 0:
                max_center_index = -1
            else:
                max_center_index = int(unique_centers.max())
            ghost_to_real_index = torch.full(
                [
                    max_center_index + 1,
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
        species.append(system.types[system_selected_atoms])

        centers.append(centers_values + node_counter)
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
        assume_unique=True,
    )

    return (
        positions,
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
    cutoff_function: str,
    cutoff_width: float,
    num_neighbors_adaptive: Optional[float] = None,
    selected_atoms: Optional[Labels] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Labels,
]:
    """
    Converts a list of systems to a batch required for the PET model.

    :param systems: List of systems to convert to a batch.
    :param options: Options for the neighbor list.
    :param all_species_list: List of all atomic species in the dataset.
    :param species_to_species_index: Mapping from atomic species to species indices.
    :param cutoff_function: Type of the smoothing function at the cutoff.
    :param cutoff_width: Width of the cutoff function for a cutoff mask.
    :param num_neighbors_adaptive: Optional maximum number of neighbors per atom.
        If provided, the adaptive cutoff scheme will be used for each atom to
        approximately select this number of neighbors.
    :param selected_atoms: Optional labels of selected atoms to include in the batch.
    :return: A tuple containing the batch tensors.
        The batch consists of the following tensors:
        - `element_indices_nodes`: The atomic species of the central atoms
        - `element_indices_neighbors`: The atomic species of the neighboring atoms
        - `edge_vectors`: The cartesian edge vectors between the central atoms and their
            neighbors
        - `edge_distances`: The distances between the central atoms and their neighbors
        - `padding_mask`: A padding mask indicating which neighbors are real, and which
            are padded
        - `reverse_neighbor_index`: The reversed neighbor list for each central atom
        - `cutoff_factors`: The cutoff function values for each edge
        - `system_indices`: The system index for each atom in the batch
        - `sample_labels`: Labels indicating the system and atom indices for each atom

    """
    (
        positions,
        centers,
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
    edge_distances = torch.norm(edge_vectors, dim=-1) + 1e-15

    if selected_atoms is not None:
        if torch.numel(centers) == 0:
            num_nodes = 0
        else:
            num_nodes = int(centers.max()) + 1
    else:
        num_nodes = len(positions)

    atomic_cutoffs = options.cutoff * torch.ones(
        num_nodes, device=positions.device, dtype=positions.dtype
    )

    if num_neighbors_adaptive is not None:
        # Enabling the adaptive cutoff scheme to approximately select
        # `num_neighbors_adaptive` neighbors for each atom

        with torch.profiler.record_function("PET::get_adaptive_cutoffs"):
            adapted_atomic_cutoffs = get_adaptive_cutoffs(
                centers,
                edge_distances,
                num_neighbors_adaptive,
                num_nodes,
                options.cutoff,
                cutoff_width=cutoff_width,
            )

        with torch.profiler.record_function("PET::adaptive_cutoff_masking"):
            unique_centers = torch.unique(centers)
            atomic_cutoffs[unique_centers] = adapted_atomic_cutoffs[unique_centers]

            cutoff_mask = edge_distances <= adapted_atomic_cutoffs[centers]
            centers = centers[cutoff_mask]
            neighbors = neighbors[cutoff_mask]
            edge_vectors = edge_vectors[cutoff_mask]
            cell_shifts = cell_shifts[cutoff_mask]
    else:
        atomic_cutoffs = options.cutoff * torch.ones(
            len(positions), device=positions.device, dtype=positions.dtype
        )

    num_neighbors = torch.bincount(centers)

    # uncomment these to print out stats on the adaptive cutoff behavior
    # print("adaptive_cutoffs", *atomic_cutoffs.tolist())
    # print("num_neighbors", *num_neighbors.tolist())

    if num_neighbors.numel() == 0:  # no edges
        max_edges_per_node = 0
    else:
        max_edges_per_node = int(torch.max(num_neighbors))

    # Convert to NEF (Node-Edge-Feature) format:
    nef_indices, nef_to_edges_neighbor, nef_mask = get_nef_indices(
        centers, num_nodes, max_edges_per_node
    )

    # Element indices
    element_indices_nodes = species_to_species_index[species]
    element_indices_neighbors = element_indices_nodes[neighbors]

    # Send everything to NEF:
    edge_vectors = edge_array_to_nef(edge_vectors, nef_indices)
    edge_distances = torch.sqrt(torch.sum(edge_vectors**2, dim=2) + 1e-15)
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
    if cutoff_function.lower() == "bump":
        # use bump switching function for adaptive cutoff
        cutoff_factors = cutoff_func_bump(
            edge_distances, atomic_cutoffs.unsqueeze(1), cutoff_width
        )
    elif cutoff_function.lower() == "cosine":
        # backward-compatible cosine swithcing for fixed cutoff
        cutoff_factors = cutoff_func_cosine(
            edge_distances, atomic_cutoffs.unsqueeze(1), cutoff_width
        )
    else:
        raise ValueError(
            f"Unknown cutoff function type: {cutoff_function}. "
            f"Supported types are 'Cosine' and 'Bump'."
        )
    cutoff_factors[~nef_mask] = 0.0

    return (
        element_indices_nodes,
        element_indices_neighbors,
        edge_vectors,
        edge_distances,
        nef_mask,
        reverse_neighbor_index,
        cutoff_factors,
        system_indices,
        sample_labels,
    )
