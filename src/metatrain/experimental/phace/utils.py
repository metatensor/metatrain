from typing import Dict, List

import torch
from metatomic.torch import NeighborListOptions, System


def systems_to_batch(
    systems: List[System], nl_options: NeighborListOptions
) -> Dict[str, torch.Tensor]:
    device = systems[0].positions.device
    positions = torch.cat([item.positions for item in systems])
    cells = torch.stack([item.cell for item in systems])
    species = torch.cat([item.types for item in systems])
    ptr = torch.tensor([0] + [len(item) for item in systems]).cumsum(0)

    edge_index_list = []
    cell_shifts_list = []
    centers_list = []
    structures_centers_list = []
    structure_pairs_list = []
    for i, system in enumerate(systems):
        nl = system.get_neighbor_list(nl_options)
        samples = nl.samples
        edge_index_item = torch.stack(
            (samples.column("first_atom"), samples.column("second_atom")), dim=1
        )
        cell_shifts_item = torch.stack(
            (
                samples.column("cell_shift_a"),
                samples.column("cell_shift_b"),
                samples.column("cell_shift_c"),
            ),
            dim=0,
        ).T
        edge_index_list.append(edge_index_item)
        cell_shifts_list.append(cell_shifts_item)
        centers_list.append(
            torch.arange(len(system.positions), device=device, dtype=torch.int32)
        )
        structures_centers_list.append(
            torch.tensor([i] * len(system.positions), device=device, dtype=torch.int32)
        )
        structure_pairs_list.append(
            torch.tensor(
                [i] * len(samples.column("first_atom")),
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
