from typing import Dict, List

import torch
from metatomic.torch import NeighborListOptions, System


def systems_to_batch(
    systems: List[System], nl_options: NeighborListOptions
) -> Dict[str, torch.Tensor]:
    """
    Convert a list of System objects to a GNN-batch-like dictionary.

    This function creates a torch-compile-friendly batch representation with
    stacked positions, cells, number of atoms per structure, atomic types,
    and center/neighbor indices.

    :param systems: List of System objects to batch
    :param nl_options: Neighbor list options to extract neighbor information
    :return: Dictionary containing batched tensors:
        - positions: stacked positions of all atoms [N_atoms, 3]
        - cells: stacked unit cells [N_structures, 3, 3]
        - species: atomic types of all atoms [N_atoms]
        - n_atoms: number of atoms per structure [N_structures]
        - cell_shifts: cell shift vectors for all pairs [N_pairs, 3]
        - centers: local atom indices within each structure [N_atoms]
        - center_indices: global center indices for all pairs [N_pairs]
        - neighbor_indices: global neighbor indices for all pairs [N_pairs]
        - structure_centers: structure index for each atom [N_atoms]
        - structure_pairs: structure index for each pair [N_pairs]
        - structure_offsets: cumulative atom offsets per structure [N_structures]
    """
    device = systems[0].positions.device

    positions_list = []
    species_list = []
    cells_list = []
    n_atoms_list: List[int] = []
    edge_index_list = []
    cell_shifts_list = []
    centers_list = []
    structures_centers_list = []
    structure_pairs_list = []

    cumulative_num_atoms = 0
    for i, system in enumerate(systems):
        n_atoms_i = len(system.positions)
        n_atoms_list.append(n_atoms_i)

        positions_list.append(system.positions)
        species_list.append(system.types)
        cells_list.append(system.cell)

        nl = system.get_neighbor_list(nl_options)
        samples = nl.samples.values
        edge_indices = samples[:, :2]  # center and neighbor indices
        cell_shifts_item = samples[:, 2:]  # cell shift vectors for periodic images

        # Create global indices by adding cumulative offset
        global_center_indices = edge_indices[:, 0] + cumulative_num_atoms
        global_neighbor_indices = edge_indices[:, 1] + cumulative_num_atoms

        edge_index_list.append(
            torch.stack([global_center_indices, global_neighbor_indices], dim=1)
        )
        cell_shifts_list.append(cell_shifts_item)

        centers_list.append(torch.arange(n_atoms_i, device=device, dtype=torch.int32))
        structures_centers_list.append(
            torch.full((n_atoms_i,), i, device=device, dtype=torch.int32)
        )
        structure_pairs_list.append(
            torch.full((len(edge_indices),), i, device=device, dtype=torch.int32)
        )

        cumulative_num_atoms += n_atoms_i

    positions = torch.cat(positions_list, dim=0)
    species = torch.cat(species_list, dim=0)
    cells = torch.stack(cells_list, dim=0)
    n_atoms = torch.tensor(n_atoms_list, device=device, dtype=torch.int64)
    pairs = torch.cat(edge_index_list, dim=0)
    cell_shifts = torch.cat(cell_shifts_list, dim=0)
    centers = torch.cat(centers_list, dim=0)
    structure_centers = torch.cat(structures_centers_list, dim=0)
    structure_pairs = torch.cat(structure_pairs_list, dim=0)

    # Compute structure offsets (cumulative sum of n_atoms, starting with 0)
    structure_offsets = torch.zeros(len(systems), device=device, dtype=torch.int32)
    if len(systems) > 1:
        structure_offsets[1:] = torch.cumsum(n_atoms[:-1].to(torch.int32), dim=0)

    batch_dict = {
        "positions": positions,
        "cells": cells,
        "species": species,
        "n_atoms": n_atoms,
        "cell_shifts": cell_shifts,
        "centers": centers,
        "center_indices": pairs[:, 0],
        "neighbor_indices": pairs[:, 1],
        "structure_centers": structure_centers,
        "structure_pairs": structure_pairs,
        "structure_offsets": structure_offsets,
    }
    return batch_dict
