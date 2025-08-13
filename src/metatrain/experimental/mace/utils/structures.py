from typing import List

import torch
from metatomic.torch import NeighborListOptions, System

from mace.data import AtomicData
from mace.tools import torch_geometric

def concatenate_structures(
    systems: List[System], neighbor_list_options: NeighborListOptions
):
    positions = []
    centers = []
    neighbors = []
    species = []
    cell_shifts = []
    cells = []
    node_counter = 0

    for system in systems:
        positions.append(system.positions)
        species.append(system.types)

        assert len(system.known_neighbor_lists()) >= 1, "no neighbor list found"
        neighbor_list = system.get_neighbor_list(neighbor_list_options)
        nl_values = neighbor_list.samples.values

        centers.append(nl_values[:, 0] + node_counter)
        neighbors.append(nl_values[:, 1] + node_counter)
        cell_shifts.append(nl_values[:, 2:])

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

def create_batch(
    systems: List[System], 
    neighbor_list_options: NeighborListOptions,
    atomic_types_to_species_index: torch.Tensor,
    n_types: int,  # Mapping from atomic types to species index
):
    """
    Create a batch of systems by concatenating their structures.

    :param systems: List of System objects to concatenate.
    :param neighbor_list_options: Options for the neighbor list.
    :return: Concatenated positions, centers, neighbors, species, cells, and cell shifts.
    """

    all_data = []

    for system in systems:

        assert len(system.known_neighbor_lists()) >= 1, "no neighbor list found"
        neighbor_list = system.get_neighbor_list(neighbor_list_options)
        nl_values = neighbor_list.samples.values

        data = AtomicData(
            node_attrs=torch.nn.functional.one_hot(atomic_types_to_species_index[system.types], num_classes=n_types).type(torch.double),  # [n_nodes, n_species]
            positions=system.positions,
            edge_index=nl_values[:, :2].T.type(torch.long),  # [2, n_edges]
            shifts=nl_values[:, 2:], # This should be multiplied by the cell
            unit_shifts=nl_values[:, 2:],  # [n_edges, 3]
            cell=system.cell,  # [3,3]
            weight=None,  # [,]
            head=None,  # [,]
            energy_weight=None,  # [,]
            forces_weight=None,  # [,]
            stress_weight=None,  # [,]
            virials_weight=None,  # [,]
            dipole_weight=None,
            charges_weight=None,
            forces=None,  # [n_nodes, 3]
            energy=None,  # [, ]
            stress=None,  # [1,3,3]
            virials=None,  # [1,3,3]
            dipole=None,  # [, 3]
            charges=None,  # [n_nodes, ]     
        )

        all_data.append(data)

    loader = torch_geometric.dataloader.DataLoader(
        dataset=all_data,
        batch_size=len(all_data),
        shuffle=False,
        drop_last=False,
    )

    return next(iter(loader))

def create_batch(
    systems: List[System],
    neighbor_list_options: NeighborListOptions,
    atomic_types_to_species_index: torch.Tensor,
    n_types: int,  # Mapping from atomic types to species index
    device: torch.device
) -> dict[str, torch.Tensor]:
    unit_shifts = []
    cell_shifts = []
    edge_index = []
    atom_types = []
    batch = []
    system_start_index = [0]

    dtype = systems[0].positions.dtype

    for system_i, system in enumerate(systems):
        neighbors = system.get_neighbor_list(neighbor_list_options)
        start_index = system_start_index[-1]

        # TODO: make this faster?
        atom_types.append(
            atomic_types_to_species_index[system.types] 
        )

        shifts = neighbors.samples.view(
            ["cell_shift_a", "cell_shift_b", "cell_shift_c"]
        ).values

        unit_shifts.append(shifts)
        cell_shifts.append(shifts.to(dtype) @ system.cell)
        edge_index.append(
            neighbors.samples.view(["first_atom", "second_atom"]).values.T.to(
                torch.int64
            )
            + start_index
        )

        n_atoms = len(system)
        batch.append(torch.full((n_atoms,), system_i))
        system_start_index.append(start_index + n_atoms)

    return {
        "positions": torch.vstack([s.positions for s in systems]),
        "cell": torch.vstack([s.cell for s in systems]),
        "unit_shifts": torch.vstack(unit_shifts).T,
        "edge_index": torch.hstack(edge_index),
        "shifts": torch.vstack(cell_shifts),
        "head": torch.tensor([0] * len(systems)).to(device),
        "batch": torch.hstack(batch).to(device),
        "ptr": torch.tensor(system_start_index).to(device),
        "node_attrs": torch.nn.functional.one_hot(
            torch.hstack(atom_types), num_classes=n_types
        ).to(dtype),
    }