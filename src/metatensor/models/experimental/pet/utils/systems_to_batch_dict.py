from typing import Dict, List, Optional, Tuple

import torch
from metatensor.torch import Labels
from metatensor.torch.atomistic import NeighborListOptions, System


def collate_graph_dicts(
    graph_dicts: List[Dict[str, torch.Tensor]]
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
    return int(torch.stack(max_system_num_neighbors).max().item())


def get_central_species(
    system: System, all_species: torch.Tensor, unique_index: torch.Tensor
) -> torch.Tensor:
    """
    Returns the indices of the species of the central atoms in the system
    in a list of all species.

    """
    species = system.types[unique_index]
    return torch.where(all_species.unsqueeze(1) == species)[0]


def get_system_batch_dict(
    system: System,
    options: NeighborListOptions,
    all_species: torch.Tensor,
    max_num_neighbors: int,
    selected_atoms_index: torch.Tensor,
    device: torch.device,
):

    nl = system.get_neighbor_list(options)
    i_list = nl.samples.column("first_atom")
    j_list = nl.samples.column("second_atom")

    unique_neighbors_index, counts = torch.unique(i_list, return_counts=True)
    unique_index = torch.unique(
        torch.cat((selected_atoms_index, unique_neighbors_index))
    )

    i_list, j_list = remap_to_contiguous_indexing(i_list, j_list, unique_index, device)

    central_species = get_central_species(system, all_species, unique_index)

    index = torch.argsort(i_list, stable=True)
    j_list = nl.samples.column("second_atom")[index]
    # This calcualtes, how many neighbors each central atom has

    # get_number_of_neighbors
    number_of_neighbors = torch.zeros(len(system), device=device, dtype=torch.int64)
    number_of_neighbors[unique_neighbors_index] = counts

    # This calculates the cumulative sum of the counts to slice
    # the index tensor and add indices of the neighbors to the final tensor
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0]), cum_sum))
    # This calculates the species of all the neighbors

    S_list: torch.Tensor = torch.cat(
        (
            nl.samples.column("cell_shift_a")[None],
            nl.samples.column("cell_shift_b")[None],
            nl.samples.column("cell_shift_c")[None],
        )
    ).transpose(0, 1)[index]

    # This calculates the indices of neighbor species in the
    # all_species tensor
    neighbors_index = torch.zeros(
        (len(system), max_num_neighbors), device=device, dtype=torch.int64
    )
    neighbors_shifts = torch.zeros(
        (len(system), max_num_neighbors, 3), device=device, dtype=torch.int64
    )
    displacement_vectors = torch.zeros(
        (len(system), max_num_neighbors, 3), device=device, dtype=torch.float32
    )
    padding_mask = torch.zeros(
        (len(system), max_num_neighbors), device=device, dtype=torch.bool
    )
    for j, count in enumerate(counts):
        # For each atom, we put the neighbors species indices up
        # to the number of neighbors, while the rest of the indices
        # are just padded with the padding_value.
        neighbors_index[j, :count] = j_list[cum_sum[j] : cum_sum[j + 1]]
        neighbors_shifts[j, :count] = S_list[cum_sum[j] : cum_sum[j + 1]]
        displacement_vectors[j, :count] = nl.values[:, :, 0][
            cum_sum[j] : cum_sum[j + 1]
        ]
        padding_mask[j, count:] = True

    # get_neighbor_species
    neighbor_species = central_species[neighbors_index]

    # get_reversed_neighbor_species
    reversed_neighbors_index = torch.zeros_like(neighbors_index)
    tmp_reversed_index = neighbors_index[neighbors_index]
    tmp_reversed_shifts = neighbors_shifts[neighbors_index]
    for j in range(len(system)):
        condition_1 = tmp_reversed_index[j] == j
        condition_2 = torch.all(
            tmp_reversed_shifts[j] == -neighbors_shifts[j].unsqueeze(1), dim=2
        )
        condition = condition_1 & condition_2
        tmp_index_1, tmp_index_2 = torch.where(condition)
        if len(tmp_index_1) > 0:
            _, counts = torch.unique(tmp_index_1, return_counts=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))
            reversed_neighbors_index[j, : number_of_neighbors[j]] = tmp_index_2[
                cum_sum
            ][: number_of_neighbors[j]]
    system_dict = {
        "central_species": central_species,
        "x": displacement_vectors,
        "neighbor_species": neighbor_species,
        "neighbors_pos": reversed_neighbors_index,
        "neighbors_index": neighbors_index,
        "nums": number_of_neighbors,
        "mask": padding_mask,
    }
    return system_dict


def systems_to_batch_dict(
    systems: List[System],
    options: NeighborListOptions,
    all_species_list: List[int],
    selected_atoms: Optional[Labels] = None,
) -> Dict[str, torch.Tensor]:
    """
    Converts a standatd input data format of `metatensor-models` to a
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


def remap_to_contiguous_indexing(
    i_list: torch.Tensor,
    j_list: torch.Tensor,
    unique_index: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    index_map: Dict[int, int] = {int(index): i for i, index in enumerate(unique_index)}
    i_list = torch.tensor(
        [index_map[int(index)] for index in i_list],
        dtype=i_list.dtype,
        device=device,
    )
    j_list = torch.tensor(
        [index_map[int(index)] for index in j_list],
        dtype=j_list.dtype,
        device=device,
    )
    return i_list, j_list
