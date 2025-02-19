from typing import Dict, List, Optional, Tuple

import pet_neighbors_convert  # noqa: F401
import torch
from metatensor.torch import Labels
from metatensor.torch.atomistic import NeighborListOptions, System


def collate_graph_dicts(
    graph_dicts: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Collates a list of graphs into a single graph.

    :param graph_dicts: A list of graphs to be collated.

    :return: The collated grap (batch).
    """
    device = graph_dicts[0]["x"].device
    simple_concatenate_keys: List[str] = [
        "central_species",
        "x",
        "neighbor_species",
        "neighbors_pos",
        "nums",
        "mask",
    ]
    cumulative_adjust_keys: List[str] = ["neighbors_index"]

    result: Dict[str, List[torch.Tensor]] = {}

    n_nodes_cumulative: int = 0

    number_of_graphs: int = int(len(graph_dicts))

    for index in range(number_of_graphs):
        graph: Dict[str, torch.Tensor] = graph_dicts[index]

        for key in simple_concatenate_keys:
            if key not in result:
                result[key] = [graph[key]]
            else:
                result[key].append(graph[key])

        for key in cumulative_adjust_keys:
            if key not in result:
                graph_key: torch.Tensor = graph[key]

                now: List[torch.Tensor] = [graph_key + n_nodes_cumulative]
                result[key] = now

            else:
                graph_key_2: torch.Tensor = graph[key]

                now_2: torch.Tensor = graph_key_2 + n_nodes_cumulative
                result[key].append(now_2)

        n_atoms: int = graph["central_species"].shape[0]

        index_repeated: torch.Tensor = torch.LongTensor([index for _ in range(n_atoms)])
        if "batch" not in result.keys():
            result["batch"] = [index_repeated]
        else:
            result["batch"].append(index_repeated)

        n_nodes_cumulative += n_atoms

    result_final: Dict[str, torch.Tensor] = {}
    for key in simple_concatenate_keys + cumulative_adjust_keys:
        now_3: List[torch.Tensor] = []
        for el in result[key]:
            now_3.append(el)

        result_final[key] = torch.cat(now_3, dim=0)

    result_final["batch"] = torch.cat(result["batch"], dim=0).to(device)
    return result_final


def get_max_num_neighbors(systems: List[System], options: NeighborListOptions):
    """
    Calculates the maximum number of neighbors that atoms in a list of systems have.

    """
    max_system_num_neighbors = []
    for system in systems:
        nl = system.get_neighbor_list(options)
        i_list = nl.samples.column("first_atom")
        if len(i_list) == 0:
            max_atom_num_neighbors = torch.tensor(
                0, device=i_list.device, dtype=i_list.dtype
            )
        else:
            max_atom_num_neighbors = torch.bincount(i_list).max()
        max_system_num_neighbors.append(max_atom_num_neighbors)
    return torch.stack(max_system_num_neighbors).max()


def get_central_species(
    system: System, all_species: torch.Tensor, unique_index: torch.Tensor
) -> torch.Tensor:
    """
    Returns the indices of the species of the central atoms in the system
    in a list of all species.

    """
    species = system.types[unique_index]
    tmp_index_1, tmp_index_2 = torch.where(all_species.unsqueeze(1) == species)
    index = torch.argsort(tmp_index_2)
    return tmp_index_1[index]


def write_system_data(
    system: System,
    options: NeighborListOptions,
    selected_atoms_index: torch.Tensor,
):
    nl = system.get_neighbor_list(options)
    i_list = nl.samples.column("first_atom")
    j_list = nl.samples.column("second_atom")
    S_list = torch.cat(
        (
            nl.samples.column("cell_shift_a")[None],
            nl.samples.column("cell_shift_b")[None],
            nl.samples.column("cell_shift_c")[None],
        )
    ).transpose(0, 1)
    D_list = nl.values[:, :, 0]
    positions = system.positions
    types = system.types
    cell = system.cell
    torch.save(
        {
            "i_list": i_list,
            "j_list": j_list,
            "S_list": S_list,
            "D_list": D_list,
            "positions": positions,
            "types": types,
            "cell": cell,
            "selected_atoms_index": selected_atoms_index,
        },
        "system_data.pt",
    )


def write_batch_dict(batch_dict: Dict[str, torch.Tensor]):
    torch.save(batch_dict, "batch_dict.pt")


def remap_to_contiguous_indexing(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    unique_neighbors_index: torch.Tensor,
    unique_index: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    This helper function remaps the indices of center and neighbor atoms
    from arbitrary indexing to contgious indexing, i.e.

    from
    0, 1, 2, 54, 55, 56
    to
    0, 1, 2, 3, 4, 5.

    This remapping is required by internal implementation of PET neighbor lists, where
    indices of the atoms cannot exceed the total amount of atoms in the system.

    Shifted indices come from LAMMPS neighborlists in the case of domain decomposition
    enabled, since they contain not only the atoms in the unit cell, but also so-called
    ghost atoms, which may have a different indexing. Thus, to avoid further errors, we
    remap the indices to a contiguous format.

    """
    index_map = torch.empty(
        int(unique_index.max().item()) + 1, dtype=torch.int64, device=device
    )
    index_map[unique_index] = torch.arange(len(unique_index), device=device)
    i_list = index_map[i_list]
    j_list = index_map[j_list]
    unique_neighbors_index = index_map[unique_neighbors_index]
    return i_list, j_list, unique_neighbors_index


def get_system_batch_dict(
    system: System,
    options: NeighborListOptions,
    all_species: torch.Tensor,
    max_num_neighbors: torch.Tensor,
    selected_atoms_index: torch.Tensor,
    device: torch.device,
    debug: bool = False,
) -> Dict[str, torch.Tensor]:
    if debug:
        write_system_data(system, options, selected_atoms_index)
    nl = system.get_neighbor_list(options)
    i_list = nl.samples.column("first_atom")
    j_list = nl.samples.column("second_atom")

    unique_neighbors_index, _ = torch.unique(i_list, return_counts=True)
    unique_index = torch.unique(
        torch.cat((selected_atoms_index, unique_neighbors_index))
    )

    # We calculate the actual size of the system, which is the number of
    # unique atoms in the system.
    # This is required for LAMMPS interface, because by default
    # it produces the system with both local and ghost atoms.
    actual_system_size = len(unique_index)

    # Then we remap the indices of the atoms to a contiguous format.
    # Also see the docstring of the function for more details.
    i_list, j_list, unique_neighbors_index = remap_to_contiguous_indexing(
        i_list, j_list, unique_neighbors_index, unique_index, device
    )

    # We sort the indices of the atoms in the system, to join the
    # periodic images of the same atom together. Otherwise, the
    # neighbor list may have a discontinuous indexing, like:
    # >>> i_list
    # tensor([0, 1, 2, 0, 1, 2])
    # instead of
    # >>> i_list
    # tensor([0, 0, 1, 1, 2, 2])
    # and we heavily rely on the fact that the indices of the atoms
    # are contiguous below.
    index = torch.argsort(i_list, stable=True)
    j_list = j_list[index].contiguous()
    i_list = i_list[index].contiguous()
    S_list: torch.Tensor = (
        torch.cat(
            (
                nl.samples.column("cell_shift_a")[None],
                nl.samples.column("cell_shift_b")[None],
                nl.samples.column("cell_shift_c")[None],
            )
        )
        .transpose(0, 1)[index]
        .to(torch.int64)
    ).contiguous()
    D_list = (
        system.positions[j_list]
        - system.positions[i_list]
        + S_list.to(system.cell.dtype) @ system.cell
    ).contiguous()

    species = system.types[unique_index].to(torch.int64)

    res = torch.ops.neighbors_convert.process(
        i_list,
        j_list,
        S_list,
        D_list,
        max_num_neighbors,
        torch.tensor(actual_system_size),
        species,
        all_species,
    )

    # `neighbors_index`: indices of the neighbors
    # from j_list for each central atom
    neighbors_index = res[0]
    # `relative_positions`: displacements of the neighbors
    # from j_list relative to each central atom
    relative_positions = res[1]
    # `nums`: number of neighbors for each central atom
    nums = res[2]
    # `mask`: for each central atom extracts the actual
    # neighbors data from the remaining tensors (i.e neighbors_index,
    # relative_positions, etc). The reason for this, is a padding that
    # enlarges the leghts of these tensors for each central atom up to
    # maximum number of neighbors across all the central atoms, to fix
    # the dimensionality of the neighborlist and fit is into a single
    # tensor. Thus, mask essentially show, where the padding is used, and
    # where is the actual NL data.
    mask = res[3]
    # `neighbor_species`: the indices of the species of the neighbors in the
    # all_species tensor.
    neighbor_species = res[4]
    # `neighbors_pos`: the reversed neighbors index, which is used in the
    # PET model to account for edge information update not only with the central
    # atoms data, but also with the neighbors data. This requires knowing the
    # reversed indices of the neighbors in the neighbor list.
    #
    # The reversed neighbor index is basically the index of the central atom in the
    # neighbor list of the neighbor atom.
    #
    # Example:
    # >>> neighbors_index
    # tensor([[25, 28, 39, ...],
    #         ...
    #         [ 2,  3,  4, ...]])
    # >>> neighbors_pos # reversed_neighbors_index
    # tensor([[ 3,  4,  1, ...],
    #         ...
    #         [ 3,  8,  7, ...]])
    #
    # The first atom has the neighbors with indices 25, 28, 39, etc.,
    # That means, in the list of neighbors of 25th atom, 4th atom will
    # have the index 0 (i.e. that will be the first atom).
    #
    # >>> neighbors_index[25]
    # tensor([45, 29, 47,  0, ...])
    # >>> neighbors_index[25][3]
    # tensor(0)
    #
    # and this is the element [0, 0] of the `neighbors_pos`.
    #
    # We also demand the reversed cell shift vector to be the opposite
    # of the original cell shift vector. This is because sometimes the
    # central atom may have two neighbors, which are the same atom, but
    # different periodic images.
    neighbors_pos = res[5]
    # central_species: the indices of species of the central
    # atoms of the system in the all_species tensor.
    central_species = res[6]

    system_dict = {
        "central_species": central_species,
        "neighbor_species": neighbor_species,
        "x": relative_positions,
        "neighbors_pos": neighbors_pos,
        "neighbors_index": neighbors_index,
        "nums": nums,
        "mask": mask,
    }
    if debug:
        write_batch_dict(system_dict)
    return system_dict


def systems_to_batch_dict(
    systems: List[System],
    options: NeighborListOptions,
    all_species_list: List[int],
    selected_atoms: Optional[Labels] = None,
) -> Dict[str, torch.Tensor]:
    """
    Converts a standard input data format of `metatrain` to a
    PyTorch Geometric `Batch` object, compatible with `PET` model.

    :param systems: The list of systems in `metatensor.torch.atomistic.System`
    format, that needs to be converted.
    :param options: A `NeighborListOptions` objects specifying the parameters
    for a neighbor list, which will be used during the convertation.
    :param all_species: A `torch.Tensor` with all the species present in the
    systems.

    :return: Batch compatible with PET.
    """
    device = systems[0].positions.device
    all_species = torch.tensor(all_species_list, device=device)
    batch: List[Dict[str, torch.Tensor]] = []
    max_num_neighbors = get_max_num_neighbors(systems, options)
    for i, system in enumerate(systems):
        if selected_atoms is not None:
            selected_atoms_index = selected_atoms.values[:, 1][
                selected_atoms.values[:, 0] == i
            ]
        else:
            selected_atoms_index = torch.arange(len(system), device=device)
        system_dict = get_system_batch_dict(
            system,
            options,
            all_species,
            max_num_neighbors,
            selected_atoms_index,
            device,
        )
        batch.append(system_dict)
    return collate_graph_dicts(batch)
