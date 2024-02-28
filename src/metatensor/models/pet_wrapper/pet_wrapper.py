import numpy as np
import torch
from metatensor.torch.atomistic import NeighborsListOptions
from pet.molecule import NeighborIndexConstructor, batch_to_dict
from torch_geometric.data import Batch, Data


def systems_to_pyg_graphs(systems, options, all_species):
    neighbor_index_constructors = []
    for system in systems:
        neighbors = system.get_neighbors_list(options)

        i_list = neighbors.samples["first_atom"]
        j_list = neighbors.samples["second_atom"]

        # print(i_list.shape)
        # print(j_list.shape)

        S_list = [
            neighbors.samples["cell_shift_a"][None],
            neighbors.samples["cell_shift_b"][None],
            neighbors.samples["cell_shift_c"][None],
        ]
        S_list = torch.cat(S_list)
        S_list = S_list.transpose(0, 1)
        # print(S_list.shape)
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
            np.where(all_species == specie)[0][0] for specie in np.array(system.species)
        ]
        central_species = torch.LongTensor(central_species).to(device)

        kwargs = {
            "central_species": central_species,
            "x": relative_positions,
            "neighbor_species": neighbor_species,
            "neighbors_pos": neighbors_pos,
            "neighbors_index": neighbors_index.transpose(0, 1),
            "nums": nums,
            "mask": mask,
            "n_atoms": len(system.species),
        }

        graph_now = Data(**kwargs)
        graphs.append(graph_now)
    batch = Batch.from_data_list(graphs)
    return batch


class PETMetatensorWrapper(torch.nn.Module):
    def __init__(self, pet_model, all_species):
        super(PETMetatensorWrapper, self).__init__()
        self.pet_model = pet_model
        self.all_species = all_species

    def forward(self, systems):
        options = NeighborsListOptions(
            model_cutoff=self.pet_model.hypers.R_CUT, full_list=True
        )
        batch = systems_to_pyg_graphs(systems, options, self.all_species)
        # print(batch_to_dict(batch))
        return self.pet_model(batch_to_dict(batch))