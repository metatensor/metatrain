from typing import Dict, List

import torch
from metatensor.torch import TensorBlock, TensorMap
from metatomic.torch import System


def is_already_per_atom(block: TensorBlock) -> bool:
    """Whether the values of a block already refer to single atoms or atom pairs.

    Such blocks (atomic energies, position gradients, atom-pair targets) are not
    divided by the number of atoms when a target is compared per atom.

    :param block: The block to inspect.

    :return: Whether the block is already per atom.
    """
    return "atom" in block.samples.names or "first_atom" in block.samples.names


def average_by_num_atoms(
    tensor_map_dict: Dict[str, TensorMap],
    systems: List[System],
    per_structure_keys: List[str],
) -> Dict[str, TensorMap]:
    """
    Averages a dictionary of ``TensorMap`` objects by the number of
    atoms in each system.

    This function averages by the number of atoms in each system. Targets that
    are present in ``per_structure_keys`` will not be averaged.

    :param tensor_map_dict: A dictionary of ``TensorMap`` objects.
    :param systems: The systems used to compute the predictions.
    :param per_structure_keys: A list of keys whose corresponding
        ``TensorMap`` objects that should not be averaged.

    :return: The dictionary of averaged ``TensorMap`` objects.
    """
    averaged_tensor_map_dict = {}
    device = systems[0].device
    num_atoms = torch.tensor([len(s) for s in systems], device=device)
    for key in tensor_map_dict.keys():
        if key in per_structure_keys:
            averaged_tensor_map_dict[key] = tensor_map_dict[key]
        else:
            averaged_tensor_map_dict[key] = divide_by_num_atoms(
                tensor_map_dict[key], num_atoms
            )
    return averaged_tensor_map_dict


def divide_by_num_atoms(tensor_map: TensorMap, num_atoms: torch.Tensor) -> TensorMap:
    """Takes the average values per atom of a ``TensorMap``.

    Since some quantities might already be per atom (e.g., atomic energies or position
    gradients) or per atom-pair (e.g. atom-pair targets), this function only divides a
    block (or gradient block) by the number of atoms if the block's samples do not
    contain the "atom" or "first_atom" key. In practice, this guarantees the desired
    behavior for the majority of the cases, including energies, forces, and virials,
    where the energies and virials should be divided by the number of atoms, while the
    forces should not.

    :param tensor_map: The input tensor map.
    :param num_atoms: The number of atoms in each system.

    :return: A new tensor map with the values divided by the number of atoms.
    """

    blocks = []
    for block in tensor_map.blocks():
        if is_already_per_atom(block):
            new_block = block.copy(deep=False)
        else:
            values = block.values / num_atoms.view(
                -1, *[1] * (len(block.values.shape) - 1)
            )
            new_block = TensorBlock(
                values=values,
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
            for gradient_name, gradient in block.gradients():
                if (
                    "atom" in gradient.samples.names
                    or "first_atom" in gradient.samples.names
                ):
                    new_gradient = gradient.copy(deep=False)
                else:
                    values = gradient.values / num_atoms.view(
                        -1, *[1] * (len(gradient.values.shape) - 1)
                    )
                    new_gradient = TensorBlock(
                        values=values,
                        samples=gradient.samples,
                        components=gradient.components,
                        properties=gradient.properties,
                    )
                new_block.add_gradient(gradient_name, new_gradient)
        blocks.append(new_block)

    return TensorMap(
        keys=tensor_map.keys,
        blocks=blocks,
    )
