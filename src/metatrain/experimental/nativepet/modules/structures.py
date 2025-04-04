from typing import List, Optional, Tuple

import torch
from metatensor.torch import Labels, TensorBlock
from metatensor.torch.atomistic import NeighborListOptions, System


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
            "only one neighbor list is supported"
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
