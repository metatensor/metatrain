from typing import List

from metatensor.torch.atomistic import (
    NeighborListOptions,
    System,
    register_autograd_neighbors,
)

from metatrain.utils.data import system_to_ase
from metatrain.utils.neighbor_lists import _compute_single_neighbor_list


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
            register_autograd_neighbors(system, neighbor_list, check_consistency=False)
            system.add_neighbor_list(options, neighbor_list)

    return system
