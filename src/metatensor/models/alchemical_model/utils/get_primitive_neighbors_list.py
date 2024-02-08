from typing import Union

import ase
import torch
from metatensor.torch.atomistic import NeighborsListOptions, System
from metatensor.torch.atomistic.ase_calculator import _compute_ase_neighbors


def get_primitive_neighbors_list(
    structure: Union[ase.Atoms, System],
    model_cutoff: float = 5.0,
    full_list: bool = True,
):
    if isinstance(structure, torch.ScriptObject):
        structure = ase.Atoms(
            numbers=structure.species.numpy(),
            positions=structure.positions.detach().numpy(),
            cell=structure.cell.detach().numpy(),
            pbc=[True, True, True],
        )
    nl_options = NeighborsListOptions(model_cutoff=model_cutoff, full_list=full_list)
    nl = _compute_ase_neighbors(structure, nl_options)
    return nl, nl_options
