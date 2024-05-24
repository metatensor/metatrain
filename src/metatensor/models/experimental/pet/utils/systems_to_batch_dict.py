from typing import Dict, List, Optional, Tuple

import torch
from metatensor.torch import Labels
from metatensor.torch.atomistic import NeighborListOptions, System


class NeighborIndexConstructor:
    """From a canonical neighbor list, this function constructs neighbor
    indices that are needed for internal usage in the PET model."""

    def __init__(
        self,
        i_list: List[int],
        j_list: List[int],
        S_list: List[torch.Tensor],
        species: List[int],
    ) -> None:
        n_atoms: int = len(species)

        self.neighbors_index: List[List[int]] = []
        for _ in range(n_atoms):
            neighbors_index_now: List[int] = []
            self.neighbors_index.append(neighbors_index_now)

        self.neighbor_shift: List[List[torch.Tensor]] = []
        for _ in range(n_atoms):
            neighbor_shift_now: List[torch.Tensor] = []
            self.neighbor_shift.append(neighbor_shift_now)

        for i, j, _, S in zip(i_list, j_list, range(len(i_list)), S_list):
            self.neighbors_index[i].append(j)
            self.neighbor_shift[i].append(S)

        self.relative_positions_raw: List[List[torch.Tensor]] = [
            [] for i in range(n_atoms)
        ]
        self.neighbor_species: List[List[int]] = []
        for _ in range(n_atoms):
            now: List[int] = []
            self.neighbor_species.append(now)

        self.neighbors_pos: List[List[torch.Tensor]] = [[] for i in range(n_atoms)]

        for i, j, index, S in zip(i_list, j_list, range(len(i_list)), S_list):
            self.relative_positions_raw[i].append(torch.LongTensor([index]))
            self.neighbor_species[i].append(species[j])
            for k in range(len(self.neighbors_index[j])):
                if (self.neighbors_index[j][k] == i) and torch.equal(
                    self.neighbor_shift[j][k], -S
                ):
                    self.neighbors_pos[i].append(torch.LongTensor([k]))
        self.relative_positions: List[torch.Tensor] = []
        for chunk in self.relative_positions_raw:
            if chunk:
                self.relative_positions.append(torch.cat(chunk, dim=0))
            else:
                self.relative_positions.append(torch.empty(0, dtype=torch.long))

    def get_max_num(self) -> int:
        maximum: int = -1
        for chunk in self.relative_positions:
            if chunk.shape[0] > maximum:
                maximum = chunk.shape[0]
        return maximum

    def get_neighbors_index(self, max_num: int, atomic_types: torch.Tensor) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        nums_raw: List[int] = []
        mask_list: List[torch.Tensor] = []
        relative_positions: torch.Tensor = torch.zeros(
            [len(self.relative_positions), max_num], dtype=torch.long
        )
        neighbors_pos: torch.Tensor = torch.zeros(
            [len(self.relative_positions), max_num], dtype=torch.long
        )
        neighbors_index: torch.Tensor = torch.zeros(
            [len(self.relative_positions), max_num], dtype=torch.long
        )

        for i in range(len(self.relative_positions)):

            now: torch.Tensor = self.relative_positions[i]

            if len(now) > 0:
                relative_positions[i, : len(now)] = now
                neighbors_pos[i, : len(now)] = torch.cat(self.neighbors_pos[i], dim=0)
                neighbors_index[i, : len(now)] = torch.LongTensor(
                    self.neighbors_index[i]
                )

            nums_raw.append(len(self.relative_positions[i]))
            current_mask: torch.Tensor = torch.zeros([max_num], dtype=torch.bool)
            current_mask[len(self.relative_positions[i]) :] = True
            mask_list.append(current_mask[None, :])

        mask: torch.Tensor = torch.cat(mask_list, dim=0).to(dtype=torch.bool)

        nums: torch.Tensor = torch.LongTensor(nums_raw)

        neighbor_species: torch.Tensor = atomic_types.shape[0] * torch.ones(
            [len(self.neighbor_species), max_num], dtype=torch.long
        )
        for i in range(len(self.neighbor_species)):
            species_now: List[int] = self.neighbor_species[i]
            values_now: List[int] = [
                int(torch.where(atomic_types == specie)[0][0].item())
                for specie in species_now
            ]
            values_now_torch: torch.Tensor = torch.LongTensor(values_now)
            neighbor_species[i, : len(values_now_torch)] = values_now_torch

        return (
            neighbors_pos,
            neighbors_index,
            nums,
            mask,
            neighbor_species,
            relative_positions,
        )


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


def systems_to_batch_dict(
    systems: List[System],
    options: NeighborListOptions,
    all_types_list: List[int],
    selected_atoms: Optional[Labels] = None,
) -> Dict[str, torch.Tensor]:
    """
    Converts a standatd input data format of `metatensor-models` to a
    PyTorch Geometric `Batch` object, compatible with `PET` model.

    :param systems: The list of systems in `metatensor.torch.atomistic.System`
    format, that needs to be converted.
    :param options: A `NeighborListOptions` objects specifying the parameters
    for a neighbor list, which will be used during the convertation.
    :param atomic_types: A `torch.Tensor` with all the species present in the
    systems.

    :return: Batch compatible with PET.
    """
    device = systems[0].positions.device
    atomic_types: torch.Tensor = torch.LongTensor(all_types_list).to(device)
    neighbors_index_constructors: List[NeighborIndexConstructor] = []

    for i, system in enumerate(systems):
        known_neighbor_lists = system.known_neighbor_lists()
        if not torch.any(
            torch.tensor([known == options for known in known_neighbor_lists])
        ):
            raise ValueError(
                f"System does not have the neighbor list with the options {options}"
            )

        neighbor = system.get_neighbor_list(options)

        i_list: torch.Tensor = neighbor.samples.column("first_atom")
        j_list: torch.Tensor = neighbor.samples.column("second_atom")
        unique_neighbors_index = torch.unique(torch.cat((i_list, j_list)))

        if selected_atoms is not None:
            selected_atoms_index = selected_atoms.values[:, 1][
                selected_atoms.values[:, 0] == i
            ]
            unique_index = torch.unique(
                torch.cat((selected_atoms_index, unique_neighbors_index))
            )
        else:
            unique_index = torch.arange(len(system))

        S_list_raw: List[torch.Tensor] = [
            neighbor.samples.column("cell_shift_a")[None],
            neighbor.samples.column("cell_shift_b")[None],
            neighbor.samples.column("cell_shift_c")[None],
        ]

        S_list: torch.Tensor = torch.cat(S_list_raw)
        S_list = S_list.transpose(0, 1)

        # unique_index = torch.unique(torch.cat((i_list, j_list)))
        species: torch.Tensor = system.types[unique_index]

        # Remapping to contiguous indexing (see `remap_to_contiguous_indexing`
        # docstring)
        if (len(unique_neighbors_index) > 0) and (
            len(unique_neighbors_index) < i_list.max()
            or len(unique_neighbors_index) < j_list.max()
        ):
            i_list, j_list = remap_to_contiguous_indexing(i_list, j_list, unique_index)

        i_list = i_list.cpu()
        j_list = j_list.cpu()
        S_list = S_list.cpu()
        species = species.cpu()

        i_list_proper: List[int] = [int(el.item()) for el in i_list]
        j_list_proper: List[int] = [int(el.item()) for el in j_list]
        S_list_proper: List[torch.Tensor] = [el.to(dtype=torch.long) for el in S_list]
        species_proper: List[int] = [int(el.item()) for el in species]

        neighbors_index_constructor: NeighborIndexConstructor = (
            NeighborIndexConstructor(
                i_list_proper, j_list_proper, S_list_proper, species_proper
            )
        )
        neighbors_index_constructors.append(neighbors_index_constructor)

    max_nums: List[int] = [
        neighbors_index_constructor.get_max_num()
        for neighbors_index_constructor in neighbors_index_constructors
    ]
    max_num: int = max(max_nums)

    graphs: List[Dict[str, torch.Tensor]] = []
    for i, (neighbors_index_constructor, system) in enumerate(
        zip(neighbors_index_constructors, systems)
    ):
        (
            neighbors_pos,
            neighbors_index,
            nums,
            mask,
            neighbor_species,
            relative_positions_index,
        ) = neighbors_index_constructor.get_neighbors_index(max_num, atomic_types)

        neighbor = system.get_neighbor_list(options)
        displacement_vectors = neighbor.values[:, :, 0].to(torch.float32)
        device = str(displacement_vectors.device)
        neighbors_pos = neighbors_pos.to(device)
        neighbors_index = neighbors_index.to(device)
        nums = nums.to(device)
        mask = mask.to(device)
        neighbor_species = neighbor_species.to(device)
        relative_positions_index = relative_positions_index.to(device)
        if len(displacement_vectors) == 0:
            shape = relative_positions_index.shape
            relative_positions = torch.zeros(
                size=(shape[0], shape[1], 3),
                device=device,
                dtype=torch.float32,
            )
        else:
            relative_positions = displacement_vectors[relative_positions_index]
        if selected_atoms is not None:
            neighbor = system.get_neighbor_list(options)
            i_list = neighbor.samples.column("first_atom")
            j_list = neighbor.samples.column("second_atom")
            selected_atoms_index = selected_atoms.values[:, 1][
                selected_atoms.values[:, 0] == i
            ]
            unique_neighbors_index = torch.unique(torch.cat((i_list, j_list)))
            unique_index = torch.unique(
                torch.cat((selected_atoms_index, unique_neighbors_index))
            )
        else:
            unique_index = torch.arange(len(system))
        species = system.types[unique_index]
        central_species = [
            int(torch.where(atomic_types == specie)[0][0].item()) for specie in species
        ]

        central_species = torch.LongTensor(central_species).to(device)

        graph_now = {
            "central_species": central_species,
            "x": relative_positions,
            "neighbor_species": neighbor_species,
            "neighbors_pos": neighbors_pos,
            "neighbors_index": neighbors_index,
            "nums": nums,
            "mask": mask,
        }
        graphs.append(graph_now)
    return collate_graph_dicts(graphs)


def remap_to_contiguous_indexing(
    i_list: torch.Tensor, j_list: torch.Tensor, unique_index: torch.Tensor
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
        device=i_list.device,
    )
    j_list = torch.tensor(
        [index_map[int(index)] for index in j_list],
        dtype=j_list.dtype,
        device=j_list.device,
    )
    return i_list, j_list
