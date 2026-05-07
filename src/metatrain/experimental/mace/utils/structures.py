from typing import List

import torch
from metatomic.torch import NeighborListOptions, System


def create_batch(
    systems: List[System],
    neighbor_list_options: NeighborListOptions,
    atomic_types_to_species_index: torch.Tensor,
    n_types: int,
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
    batch = []
    system_start_index = [0]

    dtype = systems[0].positions.dtype
    device = systems[0].device

    for system_i, system in enumerate(systems):
        neighbors = system.get_neighbor_list(neighbor_list_options)
        start_index = system_start_index[-1]

        # TODO: make this faster?
        atom_types.append(atomic_types_to_species_index[system.types])

        shifts = torch.stack(
            [
                neighbors.samples.column("cell_shift_a"),
                neighbors.samples.column("cell_shift_b"),
                neighbors.samples.column("cell_shift_c"),
            ],
            dim=-1,
        )
        unit_shifts.append(shifts)
        cell_shifts.append(shifts.to(dtype) @ system.cell)

        neightbor_atoms = torch.stack(
            [
                neighbors.samples.column("first_atom"),
                neighbors.samples.column("second_atom"),
            ],
            dim=0,
        ).to(torch.int64)
        edge_index.append(neightbor_atoms + start_index)

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
