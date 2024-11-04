from typing import List, Optional
from metatensor.torch.atomistic import System
from metatensor.torch import Labels


def extract_selected_atoms(
    systems: List[System], selected_atoms: Optional[Labels] = None
):
    """
    Preprocesses the systems by selecting only the atoms in selected_atoms.
    This is particularly important for LAMMPS interface, which returns both
    real and ghost atoms as a part of the system. This happens when the
    length of the `system` is greater than the length of the `selected_atoms`.

    :param systems: List of systems to preprocess.
    :param selected_atoms: The atoms to select from the systems.
    :return: The preprocessed systems.
    """
    if selected_atoms is None:
        return systems
    processed_systems: List[System] = []
    for i, system in enumerate(systems):
        selected_atoms_index = selected_atoms.values[:, 1][
            selected_atoms.values[:, 0] == i
        ]
        if len(system) > len(selected_atoms_index):
            positions = system.positions[selected_atoms_index]
            types = system.types[selected_atoms_index]
            cell = system.cell
            pbc = system.pbc
            processed_system = System(
                positions=positions, types=types, cell=cell, pbc=pbc
            )
            for nl_option in system.known_neighbor_lists():
                nl = system.get_neighbor_list(nl_option)
                processed_system.add_neighbor_list(nl_option, nl)
        else:
            processed_system = system
        processed_systems.append(processed_system)
    return processed_systems
