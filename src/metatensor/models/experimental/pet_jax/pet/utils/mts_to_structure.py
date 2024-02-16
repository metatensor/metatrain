from collections import namedtuple

import ase.neighborlist
import numpy as np
from metatensor.torch.atomistic import System


Structure = namedtuple(
    "Structure",
    "positions, cell, numbers, centers, neighbors, cell_shifts, energy, forces",
)


def mts_to_structure(
    system: System, energy: float, forces: np.ndarray, cutoff: float
) -> Structure:
    """Converts a `metatensor.torch.atomistic.System` to a `Structure`."""
    positions = system.positions.numpy()
    numbers = system.species.numpy()
    cell = system.cell[:].numpy()

    centers, neighbors, cell_shifts = ase.neighborlist.primitive_neighbor_list(
        quantities="ijS",
        positions=system.positions.numpy(),
        cell=system.cell.numpy(),
        pbc=[not np.all(system.cell.numpy() == 0)] * 3,
        cutoff=cutoff,
        self_interaction=False,
        use_scaled_positions=False,
    )

    return Structure(
        positions=positions,
        cell=cell,
        numbers=numbers,
        centers=centers,
        neighbors=neighbors,
        cell_shifts=cell_shifts,
        energy=energy,
        forces=forces,
    )
