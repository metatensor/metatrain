from typing import List

import ase.neighborlist
import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock
from metatensor.torch.atomistic import (
    NeighborListOptions,
    System,
    register_autograd_neighbors,
)

from .data.system_to_ase import system_to_ase


def get_system_with_neighbor_lists(
    system: System, neighbor_lists: List[NeighborListOptions]
) -> System:
    """Attaches neighbor lists to a `System` object.

    :param system: The system for which to calculate neighbor lists.
    :param neighbor_lists: A list of `NeighborListOptions` objects,
        each of which specifies the parameters for a neighbor list.

    :return: The `System` object with the neighbor lists added.
    """
    # Convert the system to an ASE atoms object
    atoms = system_to_ase(system)

    # Compute the neighbor lists
    for options in neighbor_lists:
        if options not in system.known_neighbor_lists():
            neighbor_list = _compute_single_neighbor_list(atoms, options).to(
                device=system.device, dtype=system.dtype
            )
            register_autograd_neighbors(system, neighbor_list)
            system.add_neighbor_list(options, neighbor_list)

    return system


def _compute_single_neighbor_list(
    atoms: ase.Atoms, options: NeighborListOptions
) -> TensorBlock:
    # Computes a single neighbor list for an ASE atoms object
    # (as in metatensor.torch.atomistic)

    nl_i, nl_j, nl_S, nl_D = ase.neighborlist.neighbor_list(
        "ijSD",
        atoms,
        cutoff=options.cutoff,
    )

    selected = []
    for pair_i, (i, j, S) in enumerate(zip(nl_i, nl_j, nl_S)):
        # we want a half neighbor list, so drop all duplicated neighbor
        if j < i:
            continue
        elif i == j:
            if S[0] == 0 and S[1] == 0 and S[2] == 0:
                # only create pairs with the same atom twice if the pair spans more
                # than one unit cell
                continue
            elif S[0] + S[1] + S[2] < 0 or (
                (S[0] + S[1] + S[2] == 0) and (S[2] < 0 or (S[2] == 0 and S[1] < 0))
            ):
                # When creating pairs between an atom and one of its periodic
                # images, the code generate multiple redundant pairs (e.g. with
                # shifts 0 1 1 and 0 -1 -1); and we want to only keep one of these.
                # We keep the pair in the positive half plane of shifts.
                continue

        selected.append(pair_i)

    selected = np.array(selected, dtype=np.int32)
    n_pairs = len(selected)

    if options.full_list:
        distances = np.empty((2 * n_pairs, 3), dtype=np.float64)
        samples = np.empty((2 * n_pairs, 5), dtype=np.int32)
    else:
        distances = np.empty((n_pairs, 3), dtype=np.float64)
        samples = np.empty((n_pairs, 5), dtype=np.int32)

    samples[:n_pairs, 0] = nl_i[selected]
    samples[:n_pairs, 1] = nl_j[selected]
    samples[:n_pairs, 2:] = nl_S[selected]

    distances[:n_pairs] = nl_D[selected]

    if options.full_list:
        samples[n_pairs:, 0] = nl_j[selected]
        samples[n_pairs:, 1] = nl_i[selected]
        samples[n_pairs:, 2:] = -nl_S[selected]

        distances[n_pairs:] = -nl_D[selected]

    distances = torch.from_numpy(distances)
    return TensorBlock(
        values=distances.reshape(-1, 3, 1),
        samples=Labels(
            names=[
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            values=torch.from_numpy(samples),
        ),
        components=[Labels.range("xyz", 3)],
        properties=Labels.range("distance", 1),
    )
