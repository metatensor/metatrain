from typing import Optional

import torch
from metatensor.torch import Labels
from metatomic.torch import System


def get_samples_labels(
    systems: list[System],
    selected_atoms: Optional[Labels] = None,
    sample_kinds: Optional[list[str]] = None,
) -> dict[str, Labels]:
    """
    Get the sample labels for a list of systems.

    :param systems: List of systems for which to get samples.
    :param selected_atoms: Optional labels to specify the subset
        of atoms to consider for each system.
    :param sample_kinds: List of sample kinds for which to get
        labels. If None, generate labels for all sample kinds.
    :return: A dictionary containing the sample labels for the
      different sample kinds.
    """
    if sample_kinds is None:
        sample_kinds = ["system", "atom"]

    return_dict: dict[str, Labels] = {}
    num_systems = len(systems)
    device = systems[0].positions.device

    if "system" in sample_kinds:
        return_dict["system"] = Labels(
            names=["system"],
            values=torch.arange(num_systems, device=device, dtype=torch.long).reshape(
                -1, 1
            ),
            assume_unique=True,
        )

    if "atom" in sample_kinds and selected_atoms is not None:
        return_dict["atom"] = selected_atoms
    elif "atom" in sample_kinds:
        sizes: list[int] = []
        node_offsets_list: list[int] = []

        node_counter = 0
        for system in systems:
            system_size = len(system)
            node_offsets_list.append(node_counter)
            sizes.append(system_size)
            node_counter += system_size

        sizes_tensor = torch.tensor(sizes, device=device, dtype=torch.long)
        node_offsets = torch.tensor(node_offsets_list, device=device, dtype=torch.long)

        atom_system_indices = torch.repeat_interleave(
            torch.arange(num_systems, device=device),
            sizes_tensor,
            output_size=node_counter,
        )
        atom_indices = torch.arange(
            node_counter, device=device, dtype=torch.long
        ) - torch.repeat_interleave(
            node_offsets, sizes_tensor, output_size=node_counter
        )

        atom_sample_values = torch.stack([atom_system_indices, atom_indices], dim=1)
        return_dict["atom"] = Labels(
            names=["system", "atom"],
            values=atom_sample_values,
            assume_unique=True,
        )

    return return_dict
