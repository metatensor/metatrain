import torch
from torch_geometric.data import Data, Batch

from metatensor.torch.atomistic import System, NeighborsListOptions
from pet.molecule import NeighborIndexConstructor
from typing import List


def systems_to_pyg_graphs(
    systems: List[System], options: NeighborsListOptions, all_species: List[int]
) -> Batch:
    """
    Converts a standatd input data format of `metatensor-models` to a
    PyTorch Geometric `Batch` object, compatible with `PET` model.

    :param systems: The list of systems in `metatensor.torch.atomistic.System`
    format, that needs to be converted.
    :param options: A `NeighborsListOptions` objects specifying the parameters
    for a neighbor list, which will be used during the convertation.
    :param all_species: A `torch.Tensor` with all the species present in the
    systems.

    :return: The `torch_gemoetric.data.Batch` object with the neighbor lists added.
    """
    all_species = torch.LongTensor(all_species)
    neighbor_index_constructors = []
    for system in systems:
        known_neighbors_lists = system.known_neighbors_lists()
        if options not in known_neighbors_lists:
            raise ValueError(
                f"System does not have the neighbor list with the options {options}"
            )
        neighbors = system.get_neighbors_list(options)

        i_list = neighbors.samples["first_atom"]
        j_list = neighbors.samples["second_atom"]

        S_list = [
            neighbors.samples["cell_shift_a"][None],
            neighbors.samples["cell_shift_b"][None],
            neighbors.samples["cell_shift_c"][None],
        ]

        S_list = torch.cat(S_list)
        S_list = S_list.transpose(0, 1)

        species = system.species

        i_list = i_list.data.cpu().numpy()
        j_list = j_list.data.cpu().numpy()
        S_list = S_list.data.cpu().numpy()
        species = species.data.cpu().numpy()

        neighbor_index_constructor = NeighborIndexConstructor(
            i_list, j_list, S_list, species
        )
        neighbor_index_constructors.append(neighbor_index_constructor)

    max_nums = [
        neighbor_index_constructor.get_max_num()
        for neighbor_index_constructor in neighbor_index_constructors
    ]
    max_num = max(max_nums)

    graphs = []
    for neighbor_index_constructor, system in zip(neighbor_index_constructors, systems):
        (
            neighbors_pos,
            neighbors_index,
            nums,
            mask,
            neighbor_species,
            relative_positions,
        ) = neighbor_index_constructor.get_neighbor_index(max_num, all_species)

        neighbors = system.get_neighbors_list(options)
        displacement_vectors = neighbors.values[:, :, 0]

        device = displacement_vectors.device
        neighbors_pos = neighbors_pos.to(device)
        neighbors_index = neighbors_index.to(device)
        nums = nums.to(device)
        mask = mask.to(device)
        neighbor_species = neighbor_species.to(device)
        relative_positions = relative_positions.to(device)

        relative_positions = displacement_vectors[relative_positions]
        central_species = [
            torch.where(all_species == specie)[0][0] for specie in system.species
        ]
        central_species = torch.LongTensor(central_species).to(device)

        graph_now = Data(
            central_species=central_species,
            x=relative_positions,
            neighbor_species=neighbor_species,
            neighbors_pos=neighbors_pos,
            neighbors_index=neighbors_index.transpose(0, 1),
            nums=nums,
            mask=mask,
            n_atoms=len(system.species),
        )
        graphs.append(graph_now)
    batch = Batch.from_data_list(graphs)
    return batch
