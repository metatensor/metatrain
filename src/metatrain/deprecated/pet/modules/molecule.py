import ase.io
import numpy as np
import torch
from matscipy.neighbours import neighbour_list as neighbor_list
from torch_geometric.data import Data

from .long_range import get_all_k, get_reciprocal, get_volume


class Molecule:
    def __init__(
        self,
        atoms,
        r_cut,
        use_additional_scalar_attributes,
        use_long_range,
        k_cut,
        multi_target,
        target_index_key,
    ):
        self.use_additional_scalar_attributes = use_additional_scalar_attributes

        self.atoms = atoms

        positions = self.atoms.get_positions()
        species = self.atoms.get_atomic_numbers()

        self.central_species = []
        for i in range(len(positions)):
            self.central_species.append(species[i])

        if use_additional_scalar_attributes:
            scalar_attributes = self.atoms.arrays["scalar_attributes"]
            if len(scalar_attributes.shape) == 1:
                scalar_attributes = scalar_attributes[:, np.newaxis]

            self.central_scalar_attributes = scalar_attributes

        i_list, j_list, D_list, S_list = ase.neighborlist.neighbor_list(
            "ijDS", atoms, r_cut
        )

        self.neighbors_index = [[] for i in range(len(positions))]
        self.neighbors_shift = [[] for i in range(len(positions))]

        for i, j, _D, S in zip(i_list, j_list, D_list, S_list):
            self.neighbors_index[i].append(j)
            self.neighbors_shift[i].append(S)

        self.relative_positions = [[] for i in range(len(positions))]
        self.neighbor_species = [[] for i in range(len(positions))]
        self.neighbors_pos = [[] for i in range(len(positions))]

        if use_additional_scalar_attributes:
            self.neighbor_scalar_attributes = [[] for i in range(len(positions))]

        def is_same(first, second):
            for i in range(len(first)):
                if first[i] != second[i]:
                    return False
            return True

        for i, j, D, S in zip(i_list, j_list, D_list, S_list):
            self.relative_positions[i].append(D)
            self.neighbor_species[i].append(species[j])
            if use_additional_scalar_attributes:
                self.neighbor_scalar_attributes[i].append(scalar_attributes[j])
            for k in range(len(self.neighbors_index[j])):
                if (self.neighbors_index[j][k] == i) and is_same(
                    self.neighbors_shift[j][k], -S
                ):
                    self.neighbors_pos[i].append(k)

        self.use_long_range = use_long_range
        if self.use_long_range:
            self.cell = np.array(self.atoms.get_cell())
            w_1, w_2, w_3 = get_reciprocal(self.cell[0], self.cell[1], self.cell[2])
            reciprocal = np.concatenate(
                [w_1[np.newaxis], w_2[np.newaxis], w_3[np.newaxis]], axis=0
            )
            self.reciprocal = reciprocal
            self.k_vectors = get_all_k(self.cell[0], self.cell[1], self.cell[2], k_cut)
            self.k_cut = k_cut
            self.volume = get_volume(self.cell[0], self.cell[1], self.cell[2])

        if multi_target:
            self.target_index = int(self.atoms.info[target_index_key])
        else:
            self.target_index = None

    def get_max_num(self):
        maximum = None
        for chunk in self.relative_positions:
            if (maximum is None) or (len(chunk) > maximum):
                maximum = len(chunk)
        return maximum

    def get_num_k(self):
        if self.use_long_range:
            return len(self.k_vectors)
        else:
            return None

    def get_graph(self, max_num, all_species, max_num_k):
        central_species = [
            np.where(all_species == specie)[0][0] for specie in self.central_species
        ]
        central_species = torch.LongTensor(central_species)

        nums = []
        mask = []
        relative_positions = np.zeros([len(self.relative_positions), max_num, 3])
        neighbors_pos = np.zeros([len(self.relative_positions), max_num], dtype=int)
        neighbors_index = np.zeros([len(self.relative_positions), max_num], dtype=int)

        if self.use_additional_scalar_attributes:
            neighbor_scalar_attributes = np.zeros(
                [len(self.relative_positions), max_num, 1]
            )

        for i in range(len(self.relative_positions)):
            now = np.array(self.relative_positions[i])
            if len(now) > 0:
                if self.use_additional_scalar_attributes:
                    neighbor_scalar_attributes[i, : len(now)] = (
                        self.neighbor_scalar_attributes[i]
                    )
                relative_positions[i, : len(now), :] = now
                neighbors_pos[i, : len(now)] = self.neighbors_pos[i]
                neighbors_index[i, : len(now)] = self.neighbors_index[i]

            nums.append(len(self.relative_positions[i]))
            current_mask = np.zeros(max_num)
            current_mask[len(self.relative_positions[i]) :] = True
            mask.append(current_mask[np.newaxis, :])

        mask = np.concatenate(mask, axis=0)
        relative_positions = torch.tensor(
            relative_positions, dtype=torch.get_default_dtype()
        )
        nums = torch.tensor(nums, dtype=torch.get_default_dtype())
        mask = torch.BoolTensor(mask)

        neighbors_pos = torch.LongTensor(neighbors_pos)
        neighbors_index = torch.LongTensor(neighbors_index)

        neighbor_species = len(all_species) * np.ones(
            [len(self.neighbor_species), max_num], dtype=int
        )
        for i in range(len(self.neighbor_species)):
            now = np.array(self.neighbor_species[i])
            now = np.array([np.where(all_species == specie)[0][0] for specie in now])
            neighbor_species[i, : len(now)] = now
        neighbor_species = torch.LongTensor(neighbor_species)

        kwargs = {
            "central_species": central_species,
            "x": relative_positions,
            "neighbor_species": neighbor_species,
            "neighbors_pos": neighbors_pos,
            "neighbors_index": neighbors_index.transpose(0, 1),
            "nums": nums,
            "mask": mask,
            "n_atoms": len(self.atoms.positions),
        }

        if self.target_index is not None:
            kwargs["target_id"] = self.target_index

        if self.use_additional_scalar_attributes:
            kwargs["neighbor_scalar_attributes"] = torch.tensor(
                neighbor_scalar_attributes, dtype=torch.get_default_dtype()
            )
            kwargs["central_scalar_attributes"] = torch.tensor(
                self.central_scalar_attributes, dtype=torch.get_default_dtype()
            )

        if self.use_long_range:
            kwargs["cell"] = torch.tensor(self.cell, dtype=torch.get_default_dtype())[
                None
            ]
            kwargs["reciprocal"] = torch.tensor(
                self.reciprocal, dtype=torch.get_default_dtype()
            )[None]
            k_vectors = np.zeros([1, max_num_k, 3])
            k_mask = np.zeros([max_num_k], dtype=bool)
            for index in range(len(self.k_vectors)):
                k_vectors[0, index] = self.k_vectors[index]
                k_mask[index] = True
            kwargs["k_vectors"] = torch.tensor(
                k_vectors, dtype=torch.get_default_dtype()
            )
            kwargs["k_mask"] = torch.BoolTensor(k_mask)[None]

            kwargs["positions"] = torch.tensor(
                self.atoms.get_positions(), dtype=torch.get_default_dtype()
            )
            kwargs["volume"] = torch.tensor(
                self.volume, dtype=torch.get_default_dtype()
            )

        result = Data(**kwargs)

        return result


class MoleculeCPP:
    def __init__(
        self,
        atoms,
        r_cut,
        use_additional_scalar_attributes,
        use_long_range,
        k_cut,
        multi_target,
        target_index_key,
    ):
        self.use_additional_scalar_attributes = use_additional_scalar_attributes
        self.atoms = atoms
        self.r_cut = r_cut
        self.use_long_range = use_long_range
        self.k_cut = k_cut

        if self.use_long_range:
            raise NotImplementedError("Long range is not implemented in cpp")
        if self.use_additional_scalar_attributes:
            raise NotImplementedError(
                "Additional scalar attributes are not implemented in cpp"
            )

        def is_3d_crystal(atoms):
            pbc = atoms.get_pbc()
            if isinstance(pbc, bool):
                return pbc
            return all(pbc)

        if is_3d_crystal(atoms):
            i_list, j_list, D_list, S_list = neighbor_list("ijDS", atoms, r_cut)
        else:
            i_list, j_list, D_list, S_list = ase.neighborlist.neighbor_list(
                "ijDS", atoms, r_cut
            )

        self.i_list = torch.tensor(i_list, dtype=torch.int64).contiguous()
        self.j_list = torch.tensor(j_list, dtype=torch.int64).contiguous()
        self.D_list = torch.tensor(D_list, dtype=torch.get_default_dtype()).contiguous()
        self.S_list = torch.tensor(S_list, dtype=torch.int64).contiguous()
        self.species = torch.tensor(
            atoms.get_atomic_numbers(), dtype=torch.int64
        ).contiguous()
        if len(self.i_list) == 0:
            self.max_num = 0
        else:
            self.max_num = torch.max(torch.bincount(self.i_list))

        if multi_target:
            self.target_index = int(self.atoms.info[target_index_key])
        else:
            self.target_index = None

    def get_num_k(self):
        raise NotImplementedError("Long range is not implemented in cpp")

    def get_max_num(self):
        return self.max_num

    def get_graph(self, max_num, all_species, max_num_k):
        n_atoms = len(self.atoms.get_atomic_numbers())
        all_species = torch.tensor(all_species, dtype=torch.int64).contiguous()

        (
            neighbors_index,
            relative_positions,
            nums,
            mask,
            neighbor_species,
            neighbors_pos,
            species_mapped,
        ) = torch.ops.neighbors_convert.process(
            self.i_list,
            self.j_list,
            self.S_list,
            self.D_list,
            max_num,
            n_atoms,
            self.species,
            all_species,
        )

        kwargs = {
            "central_species": species_mapped,
            "x": relative_positions,
            "neighbor_species": neighbor_species,
            "neighbors_pos": neighbors_pos,
            "neighbors_index": neighbors_index.transpose(0, 1),
            "nums": nums,
            "mask": mask,
            "n_atoms": len(self.atoms.positions),
        }

        if self.target_index is not None:
            kwargs["target_id"] = self.target_index

        result = Data(**kwargs)

        return result


def batch_to_dict(batch):
    batch_dict = {
        "x": batch.x,
        "central_species": batch.central_species,
        "neighbor_species": batch.neighbor_species,
        "mask": batch.mask,
        "batch": batch.batch,
        "nums": batch.nums,
        "neighbors_index": batch.neighbors_index.transpose(0, 1),
        "neighbors_pos": batch.neighbors_pos,
    }
    if hasattr(batch, "target_id"):
        batch_dict["target_id"] = batch.target_id

    if hasattr(batch, "neighbor_scalar_attributes"):
        batch_dict["neighbor_scalar_attributes"] = batch.neighbor_scalar_attributes
    if hasattr(batch, "central_scalar_attributes"):
        batch_dict["central_scalar_attributes"] = batch.central_scalar_attributes

    if hasattr(batch, "k_vectors"):
        batch_dict["k_vectors"] = batch.k_vectors

    if hasattr(batch, "k_mask"):
        batch_dict["k_mask"] = batch.k_mask

    if hasattr(batch, "positions"):
        batch_dict["positions"] = batch.positions

    if hasattr(batch, "volume"):
        batch_dict["volume"] = batch.volume

    return batch_dict


class NeighborIndexConstructor:
    def __init__(self, i_list, j_list, S_list, species):
        n_atoms = len(species)
        self.neighbors_index = [[] for i in range(n_atoms)]
        self.neighbors_shift = [[] for i in range(n_atoms)]

        for i, j, _index, S in zip(i_list, j_list, range(len(i_list)), S_list):
            self.neighbors_index[i].append(j)
            self.neighbors_shift[i].append(S)

        self.relative_positions = [[] for i in range(n_atoms)]
        self.neighbor_species = [[] for i in range(n_atoms)]
        self.neighbors_pos = [[] for i in range(n_atoms)]

        def is_same(first, second):
            for i in range(len(first)):
                if first[i] != second[i]:
                    return False
            return True

        for i, j, index, S in zip(i_list, j_list, range(len(i_list)), S_list):
            self.relative_positions[i].append(index)
            self.neighbor_species[i].append(species[j])
            for k in range(len(self.neighbors_index[j])):
                if (self.neighbors_index[j][k] == i) and is_same(
                    self.neighbors_shift[j][k], -S
                ):
                    self.neighbors_pos[i].append(k)

    def get_max_num(self):
        maximum = None
        for chunk in self.relative_positions:
            if (maximum is None) or (len(chunk) > maximum):
                maximum = len(chunk)
        return maximum

    def get_neighbor_index(self, max_num, all_species):
        nums = []
        mask = []
        relative_positions = np.zeros([len(self.relative_positions), max_num])
        neighbors_pos = np.zeros([len(self.relative_positions), max_num], dtype=int)
        neighbors_index = np.zeros([len(self.relative_positions), max_num], dtype=int)

        for i in range(len(self.relative_positions)):
            now = np.array(self.relative_positions[i])
            if len(now) > 0:
                relative_positions[i, : len(now)] = now
                neighbors_pos[i, : len(now)] = self.neighbors_pos[i]
                neighbors_index[i, : len(now)] = self.neighbors_index[i]

            nums.append(len(self.relative_positions[i]))
            current_mask = np.zeros(max_num)
            current_mask[len(self.relative_positions[i]) :] = True
            mask.append(current_mask[np.newaxis, :])

        mask = np.concatenate(mask, axis=0)
        relative_positions = torch.LongTensor(relative_positions)
        nums = torch.tensor(nums, dtype=torch.get_default_dtype())
        mask = torch.BoolTensor(mask)

        neighbors_pos = torch.LongTensor(neighbors_pos)
        neighbors_index = torch.LongTensor(neighbors_index)

        neighbor_species = len(all_species) * np.ones(
            [len(self.neighbor_species), max_num], dtype=int
        )
        for i in range(len(self.neighbor_species)):
            now = np.array(self.neighbor_species[i])
            now = np.array([np.where(all_species == specie)[0][0] for specie in now])
            neighbor_species[i, : len(now)] = now
        neighbor_species = torch.LongTensor(neighbor_species)

        return (
            neighbors_pos,
            neighbors_index,
            nums,
            mask,
            neighbor_species,
            relative_positions,
        )
