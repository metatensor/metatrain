import ase
import numpy as np
import torch
from ase.neighborlist import primitive_neighbor_list


def get_neighbor_list(positions, pbc, cell, cutoff_radius):

    centers, neighbors, unit_cell_shift_vectors = (
        ase.neighborlist.primitive_neighbor_list(
            quantities="ijS",
            pbc=pbc,
            cell=cell,
            positions=positions,
            cutoff=cutoff_radius,
            self_interaction=True,
            use_scaled_positions=False,
        )
    )

    pairs_to_throw = np.logical_and(
        centers == neighbors, np.all(unit_cell_shift_vectors == 0, axis=1)
    )
    pairs_to_keep = np.logical_not(pairs_to_throw)

    centers = centers[pairs_to_keep]
    neighbors = neighbors[pairs_to_keep]
    unit_cell_shift_vectors = unit_cell_shift_vectors[pairs_to_keep]

    centers = torch.LongTensor(centers)
    neighbors = torch.LongTensor(neighbors)
    unit_cell_shift_vectors = torch.tensor(
        unit_cell_shift_vectors, dtype=torch.get_default_dtype()
    )

    return centers, neighbors, unit_cell_shift_vectors
