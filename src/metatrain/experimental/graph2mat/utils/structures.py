from typing import List

import sisl
import torch
from graph2mat import MatrixDataProcessor
from metatomic.torch import NeighborListOptions, System


def create_batch(
    systems: List[System],
    neighbor_list_options: NeighborListOptions,
    atomic_types_to_species_index: torch.Tensor,
    n_types: int,
    data_processor: MatrixDataProcessor,
) -> dict[str, torch.Tensor]:
    """Creates a torch geometric-like batch from a list of systems.

    The batch returned by this function can be used as input
    for MACE models.

    :param systems: List of systems to batch.
    :param neighbor_list_options: Options to create the neighbor lists.
    :param atomic_types_to_species_index: Mapping from atomic types to species index.
    :param n_types: Number of different species.

    :return: A dictionary containing the batched data.
    """
    unit_shifts = []
    cell_shifts = []
    edge_index = []
    atom_types = []
    edge_types = []
    neigh_isc = []
    batch = []
    system_start_index = [0]

    dtype = systems[0].positions.dtype
    device = systems[0].device

    for system_i, system in enumerate(systems):
        neighbors = system.get_neighbor_list(neighbor_list_options)
        start_index = system_start_index[-1]

        # TODO: make this faster?
        system_atom_types = atomic_types_to_species_index[system.types]
        atom_types.append(system_atom_types)

        shifts = neighbors.samples.view(
            ["cell_shift_a", "cell_shift_b", "cell_shift_c"]
        ).values.T

        system_edge_index = neighbors.samples.view(
            ["first_atom", "second_atom"]
        ).values.T.to(torch.int64)
        system_cell_shifts = shifts.T.to(dtype) @ system.cell

        # Get the edge types
        system_edge_types = data_processor.basis_table.point_type_to_edge_type(
            system_atom_types[system_edge_index]
        )

        # Check if there are any edges
        any_edges = system_edge_index.shape[1] > 0

        # Get the number of supercells needed along each direction to account for all interactions
        if any_edges:
            nsc = abs(shifts).max(axis=1).values * 2 + 1
        else:
            nsc = torch.tensor([1, 1, 1])

        # Then build the supercell that encompasses all of those atoms, so that we can get the
        # array that converts from sc shifts (3D) to a single supercell index. This is isc_off.
        supercell = sisl.Lattice(system.cell, nsc=nsc)

        edge_index.append(system_edge_index + start_index)
        edge_types.append(torch.from_numpy(system_edge_types).to(torch.int64))
        cell_shifts.append(system_cell_shifts)
        unit_shifts.append(shifts.T)

        # Then, get the supercell index of each interaction.
        neigh_isc.append(
            torch.tensor(supercell.isc_off[shifts[0], shifts[1], shifts[2]])
        )

        n_atoms = len(system)
        batch.append(torch.full((n_atoms,), system_i))
        system_start_index.append(start_index + n_atoms)

    return {
        "positions": torch.vstack([s.positions for s in systems]),
        "cell": torch.vstack([s.cell for s in systems]),
        "unit_shifts": torch.vstack(unit_shifts),
        "edge_index": torch.hstack(edge_index),
        "shifts": torch.vstack(cell_shifts),
        "head": torch.tensor([0] * len(systems)).to(device),
        "batch": torch.hstack(batch).to(device),
        "ptr": torch.tensor(system_start_index).to(device),
        "node_attrs": torch.nn.functional.one_hot(
            torch.hstack(atom_types), num_classes=n_types
        ).to(dtype),
        "point_types": torch.hstack(atom_types),
        "edge_types": torch.hstack(edge_types),
        "neigh_isc": torch.hstack(neigh_isc),
    }


def get_edge_vectors_and_lengths(
    positions: torch.Tensor,  # [n_nodes, 3]
    edge_index: torch.Tensor,  # [2, n_edges]
    shifts: torch.Tensor,  # [n_edges, 3]
    normalize: bool = False,
    eps: float = 1e-9,
) -> tuple[torch.Tensor, torch.Tensor]:
    sender = edge_index[0]
    receiver = edge_index[1]
    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths

    return vectors, lengths
