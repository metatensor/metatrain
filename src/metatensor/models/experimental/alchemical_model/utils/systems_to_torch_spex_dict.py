from typing import List, Optional

import torch
from metatensor.torch.atomistic import NeighborsListOptions, System


def systems_to_torch_spex_dict(
    systems: List[System], nl_options: Optional[NeighborsListOptions] = None
):
    """
    Convert a list of metatensor.torch.atomistic.Systems to a dictionary of torch
    tensors compatible with torch_spex calculators.
    """
    device = systems[0].positions.device
    positions = torch.cat([item.positions for item in systems])
    cells = torch.stack([item.cell for item in systems])
    species = torch.cat([item.species for item in systems])
    centers = torch.cat([torch.arange(len(item), device=device) for item in systems])
    if nl_options is None:
        nl_options = systems[0].known_neighbors_lists()[0]
    nls = [item.get_neighbors_list(nl_options) for item in systems]
    pairs = torch.cat(
        [
            torch.stack(
                (item.samples.column("first_atom"), item.samples.column("second_atom"))
            )
            for item in nls
        ],
        dim=1,
    ).T
    cell_shifts = torch.cat(
        [
            torch.stack(
                (
                    item.samples.column("cell_shift_a"),
                    item.samples.column("cell_shift_b"),
                    item.samples.column("cell_shift_c"),
                )
            )
            for item in nls
        ],
        dim=1,
    ).T

    lenghts = torch.tensor([len(item) for item in systems], device=device)
    nl_lenghts = torch.tensor([len(item.values) for item in nls], device=device)
    index = torch.arange(len(systems), device=device)
    system_centers = torch.repeat_interleave(index, lenghts)
    system_pairs = torch.repeat_interleave(index, nl_lenghts)
    system_offsets = torch.cat(
        [torch.tensor([0], device=device), torch.cumsum(lenghts[:-1], dim=0)]
    )

    batch_dict = {
        "positions": positions,
        "cells": cells,
        "species": species,
        "centers": centers,
        "pairs": pairs,
        "cell_shifts": cell_shifts,
        "system_centers": system_centers,
        "system_pairs": system_pairs,
        "system_offsets": system_offsets,
    }
    return batch_dict
