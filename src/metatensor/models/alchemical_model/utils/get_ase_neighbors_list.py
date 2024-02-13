from typing import Union

import ase
import torch
from metatensor.torch.atomistic import NeighborsListOptions, System
from metatensor.torch.atomistic.ase_calculator import _compute_ase_neighbors


def get_ase_neighbors_list(
    structure: Union[ase.Atoms, System],
    nl_options: NeighborsListOptions,
):
    if isinstance(structure, torch.ScriptObject):
        structure = ase.Atoms(
            numbers=structure.species.numpy(),
            positions=structure.positions.detach().numpy(),
            cell=structure.cell.detach().numpy(),
            pbc=[True, True, True],
        )
    nl = _compute_ase_neighbors(structure, nl_options)
    return nl, nl_options
