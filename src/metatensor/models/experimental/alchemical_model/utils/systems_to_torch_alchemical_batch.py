from typing import Dict, List

import torch
from metatensor.torch.atomistic import NeighborListOptions, System


def systems_to_torch_alchemical_batch(
    systems: List[System], nl_options: NeighborListOptions
) -> Dict[str, torch.Tensor]:
    """
    Convert a list of metatensor.torch.atomistic.Systems to a dictionary of torch
    tensors compatible with torch_alchemiacal calculators.
    """
    device = systems[0].positions.device
    positions = torch.cat([item.positions for item in systems])
    cells = torch.cat([item.cell for item in systems])
    numbers = torch.cat([item.types for item in systems])
    ptr = torch.tensor([0] + [len(item) for item in systems]).cumsum(0)
    batch = torch.repeat_interleave(
        torch.arange(len(systems)), torch.tensor([len(item) for item in systems])
    ).to(device)
    edge_index_list = []
    edge_offsets_list = []
    for i, system in enumerate(systems):
        nl = system.get_neighbor_list(nl_options)
        samples = nl.samples
        edge_index_item = torch.stack(
            (samples.column("first_atom"), samples.column("second_atom")), dim=0
        )
        edge_offsets_item = torch.stack(
            (
                samples.column("cell_shift_a"),
                samples.column("cell_shift_b"),
                samples.column("cell_shift_c"),
            ),
            dim=0,
        ).T
        edge_index_list.append(edge_index_item + ptr[i])
        edge_offsets_list.append(edge_offsets_item)

    edge_indices = torch.cat(edge_index_list, dim=1)
    edge_offsets = torch.cat(edge_offsets_list, dim=0)

    batch_dict = {
        "positions": positions,
        "cells": cells,
        "numbers": numbers,
        "edge_indices": edge_indices,
        "edge_offsets": edge_offsets,
        "batch": batch,
    }

    return batch_dict
