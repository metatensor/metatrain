from typing import Callable, Dict, List, Tuple

import torch
import vesin
from metatensor.torch import Labels, TensorBlock
from metatomic.torch import NeighborListOptions, System, register_autograd_neighbors


def get_system_with_neighbor_lists_transform(
    requested_neighbor_lists: List[NeighborListOptions],
) -> Callable:
    """
    Returns a function that adds the requested neighbor lists to each system.

    :param requested_neighbor_lists: A list of `NeighborListOptions` objects,
        each of which specifies the parameters for a neighbor list.
    :return: A function that takes a list of `System` objects and returns a new
        list of `System` objects with the requested neighbor lists added.
    """

    def transform(
        systems: List[System],
        targets: Dict[str, TensorBlock],
        extra: Dict[str, TensorBlock],
    ) -> Tuple[List[System], Dict[str, TensorBlock], Dict[str, TensorBlock]]:
        """
        :param systems: A list of `System` objects.
        :param targets: A dictionary of target `TensorBlock` objects.
        :param extra: A dictionary of extra `TensorBlock` objects.
        :return: The systems with the requested neighbor lists added, along with
            the original targets and extra data.
        """
        new_systems = []
        for system in systems:
            new_system = get_system_with_neighbor_lists(
                system,
                requested_neighbor_lists,
            )
            new_systems.append(new_system)
        return new_systems, targets, extra

    return transform


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
) -> None:
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
    # Compute the neighbor lists
    for options in neighbor_lists:
        if options not in system.known_neighbor_lists():
            neighbors = _compute_single_neighbor_list(system, options)

            register_autograd_neighbors(system, neighbors)
            system.add_neighbor_list(options, neighbors)

    return system


def _compute_single_neighbor_list(
    system: System, options: NeighborListOptions
) -> TensorBlock:
    # Computes a single neighbor list on the system's device (as in metatomic.torch)

    # vesin builds half lists itself, and skipping the sorting done by
    # `vesin.ase_neighbor_list` makes the calculation faster
    calculator = vesin.NeighborList(
        cutoff=float(options.cutoff),
        full_list=options.full_list,
        sorted=False,
    )

    # vesin does not support autograd, gradients are attached afterwards by
    # `register_autograd_neighbors`. Periodicity is taken from the non-zero cell
    # vectors, as was done when converting to ASE.
    nl_i, nl_j, nl_S, nl_D = calculator.compute(
        quantities="ijSD",
        points=system.positions.detach(),
        box=system.cell.detach(),
        periodic=system.cell.detach().any(dim=1),
        copy=True,
    )

    # vesin returns `i` and `j` as uint64, which torch can not promote
    samples = torch.hstack(
        [nl_i[:, None].to(torch.int32), nl_j[:, None].to(torch.int32), nl_S]
    )

    return TensorBlock(
        values=nl_D.reshape(-1, 3, 1),
        samples=Labels(
            names=[
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            values=samples,
            assume_unique=True,
        ),
        components=[Labels.range("xyz", 3).to(device=system.device)],
        properties=Labels.range("distance", 1).to(device=system.device),
    )
