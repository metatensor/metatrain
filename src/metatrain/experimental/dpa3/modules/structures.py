from typing import List, Tuple

import torch
from metatomic.torch import System


def concatenate_structures(
    systems: List[System],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate a list of Systems into padded batch tensors.

    Returns:
        positions:    [batch, max_atoms, 3]
        species:      [batch, max_atoms]  (-1 for padding)
        cells:        [batch, 3, 3]
        atom_index:   flat int32 tensor of within-system atom indices
        system_index: flat int32 tensor mapping each atom to its system
    """
    device = systems[0].positions.device
    atom_nums: List[int] = []

    atom_index_list: List[torch.Tensor] = []
    system_index_list: List[torch.Tensor] = []

    for i, system in enumerate(systems):
        atom_nums.append(len(system.positions))
        atom_index_list.append(torch.arange(start=0, end=len(system.positions)))
        system_index_list.append(torch.full((len(system.positions),), i))
    max_atom_num = max(atom_nums)
    atom_index = torch.cat(atom_index_list, dim=0).to(torch.int32).to(device)
    system_index = torch.cat(system_index_list, dim=0).to(torch.int32).to(device)

    positions = torch.zeros(
        (len(systems), max_atom_num, 3), dtype=systems[0].positions.dtype
    )
    species = torch.full((len(systems), max_atom_num), -1, dtype=systems[0].types.dtype)
    cells = torch.stack([system.cell for system in systems])  # [batch_size, 3, 3]

    for i, system in enumerate(systems):
        positions[i, : len(system.positions)] = system.positions
        species[i, : len(system.positions)] = system.types
        cells[i] = system.cell

    return (
        positions.to(device),
        species.to(device),
        cells.to(device),
        atom_index,
        system_index,
    )
