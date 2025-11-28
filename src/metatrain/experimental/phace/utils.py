from typing import Dict, List

import torch
from metatomic.torch import NeighborListOptions, System


def systems_to_list(
    systems: List[System], nl_options: NeighborListOptions
) -> List[List[torch.Tensor]]:
    systems_as_list: List[List[torch.Tensor]] = []
    for system in systems:
        nl = system.get_neighbor_list(nl_options)
        samples = nl.samples.values
        edge_indices = samples[:, :2]
        cell_shifts = samples[:, 2:]
        systems_as_list.append(
            [system.positions, system.types, system.cell, edge_indices, cell_shifts]
        )
    return systems_as_list 


def systems_to_batch(systems: List[List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
    device = systems[0][0].device
    positions = torch.cat([system[0] for system in systems])
    species = torch.cat([system[1] for system in systems])
    cells = torch.stack([system[2] for system in systems])
    ptr = torch.tensor([0] + [len(system[0]) for system in systems]).cumsum(0)

    edge_index_list = []
    cell_shifts_list = []
    centers_list = []
    structures_centers_list = []
    structure_pairs_list = []
    for i, system in enumerate(systems):
        edge_index_item = system[3]
        cell_shifts_item = system[4]
        edge_index_list.append(edge_index_item)
        cell_shifts_list.append(cell_shifts_item)
        centers_list.append(
            torch.arange(len(system[0]), device=device, dtype=torch.int32)
        )
        structures_centers_list.append(
            torch.tensor([i] * len(system[0]), device=device, dtype=torch.int32)
        )
        structure_pairs_list.append(
            torch.tensor(
                [i] * len(system[3]),
                device=device,
                dtype=torch.int32,
            )
        )

    pairs = torch.cat(edge_index_list, dim=0)
    cell_shifts = torch.cat(cell_shifts_list, dim=0)
    centers = torch.cat(centers_list, dim=0)
    structure_centers = torch.cat(structures_centers_list, dim=0)
    structure_pairs = torch.cat(structure_pairs_list, dim=0)
    structure_offsets = ptr[:-1].to(device=device, dtype=torch.int32)

    batch_dict = {
        "positions": positions,
        "cells": cells,
        "species": species,
        "cell_shifts": cell_shifts,
        "centers": centers,
        "pairs": pairs,
        "structure_centers": structure_centers,
        "structure_pairs": structure_pairs,
        "structure_offsets": structure_offsets,
    }

    return batch_dict
