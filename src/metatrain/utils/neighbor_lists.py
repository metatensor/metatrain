from typing import List

import ase.neighborlist
import numpy as np
import torch
import vesin
from metatensor.torch import Labels, TensorBlock
from metatensor.torch.atomistic import (
    NeighborListOptions,
    System,
)

from .data.system_to_ase import system_to_ase


def get_requested_neighbor_lists(
    module: torch.nn.Module,
) -> List[NeighborListOptions]:
    """Get the neighbor lists requested by a module and its children.

    :param module: The module for which to get the requested neighbor lists.

    :return: A list of `NeighborListOptions` objects requested by the module.
    """
    requested: List[NeighborListOptions] = []
    _get_requested_neighbor_lists_in_place(
        module=module,
        module_name="",
        requested=requested,
    )
    return requested


def _get_requested_neighbor_lists_in_place(
    module: torch.nn.Module,
    module_name: str,
    requested: List[NeighborListOptions],
):
    # copied from
    # metatensor/python/metatensor-torch/metatensor/torch/atomistic/model.py
    # and just removed the length units

    if hasattr(module, "requested_neighbor_lists"):
        for new_options in module.requested_neighbor_lists():
            new_options.add_requestor(module_name)

            already_requested = False
            for existing in requested:
                if existing == new_options:
                    already_requested = True
                    for requestor in new_options.requestors():
                        existing.add_requestor(requestor)

            if not already_requested:
                requested.append(new_options)

    for child_name, child in module.named_children():
        _get_requested_neighbor_lists_in_place(
            module=child,
            module_name=module_name + "." + child_name,
            requested=requested,
        )


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
            system.add_neighbor_list(options, neighbor_list)

    return system


def _compute_single_neighbor_list(
    atoms: ase.Atoms, options: NeighborListOptions
) -> TensorBlock:
    # Computes a single neighbor list for an ASE atoms object
    # (as in metatensor.torch.atomistic)

    if np.all(atoms.pbc) or np.all(~atoms.pbc):
        nl_i, nl_j, nl_S, nl_D = vesin.ase_neighbor_list(
            "ijSD",
            atoms,
            cutoff=options.cutoff,
        )
    else:
        # this is not implemented in vesin, so we use ASE
        nl_i, nl_j, nl_S, nl_D = ase.neighborlist.neighbor_list(
            "ijSD",
            atoms,
            cutoff=options.cutoff,
        )

    # The pair selection code here below avoids a relatively slow loop over
    # all pairs to improve performance
    reject_condition = (
        # we want a half neighbor list, so drop all duplicated neighbors
        (nl_j < nl_i)
        | (
            (nl_i == nl_j)
            & (
                # only create pairs with the same atom twice if the pair spans more
                # than one unit cell
                ((nl_S[:, 0] == 0) & (nl_S[:, 1] == 0) & (nl_S[:, 2] == 0))
                |
                # When creating pairs between an atom and one of its periodic images,
                # the code generates multiple redundant pairs
                # (e.g. with shifts 0 1 1 and 0 -1 -1); and we want to only keep one of
                # these. We keep the pair in the positive half plane of shifts.
                (
                    (nl_S.sum(axis=1) < 0)
                    | (
                        (nl_S.sum(axis=1) == 0)
                        & ((nl_S[:, 2] < 0) | ((nl_S[:, 2] == 0) & (nl_S[:, 1] < 0)))
                    )
                )
            )
        )
    )
    selected = np.logical_not(reject_condition)
    n_pairs = np.sum(selected)

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
