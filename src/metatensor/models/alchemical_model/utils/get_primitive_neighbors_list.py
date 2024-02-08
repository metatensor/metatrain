import ase
from metatensor.torch.atomistic import NeighborsListOptions
from metatensor.torch.atomistic.ase_calculator import _compute_ase_neighbors


def get_primitive_neighbors_list(
    structure: ase.Atoms, model_cutoff: float = 5.0, full_list: bool = True
):
    nl_options = NeighborsListOptions(model_cutoff=model_cutoff, full_list=full_list)
    nl = _compute_ase_neighbors(structure, nl_options)
    return nl, nl_options
