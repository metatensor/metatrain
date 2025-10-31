from typing import List

import torch
from metatomic.torch import NeighborListOptions, System


def create_batch(
    systems: List[System],
    neighbor_list_options: NeighborListOptions,
    atomic_types_to_species_index: torch.Tensor,
    n_types: int,  # Mapping from atomic types to species index
    device: torch.device,
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
        atom_types.append(atomic_types_to_species_index[system.types])

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
