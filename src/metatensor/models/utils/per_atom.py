from typing import Dict, List

import torch
from metatensor.torch import TensorBlock, TensorMap
from metatensor.torch.atomistic import System


def average_predictions_and_targets_by_num_atoms(
    predictions: Dict[str, TensorMap],
    targets: Dict[str, TensorMap],
    systems: List[System],
    per_structure_targets: List[str],
):
    """Averages predictions and targets by the number of atoms in each system.

    This function averages predictions and targets by the number of atoms
    in each system. Targets that are present in ``per_structure_targets`` will
    not be averaged.

    :param predictions: A dictionary of predictions.
    :param targets: A dictionary of targets.
    :param systems: The systems used to compute the predictions.
    :param per_structure_targets: A list of targets that should not be averaged.

    :return: A tuple containing the averaged predictions and targets.
    """
    averaged_predictions = {}
    averaged_targets = {}
    device = systems[0].device
    num_atoms = torch.tensor([len(s) for s in systems], device=device)
    for target in targets.keys():
        if target in per_structure_targets:
            averaged_predictions[target] = predictions[target]
            averaged_targets[target] = targets[target]
        averaged_predictions[target] = divide_by_num_atoms(
            predictions[target], num_atoms
        )
        averaged_targets[target] = divide_by_num_atoms(targets[target], num_atoms)

    return averaged_predictions, averaged_targets


def divide_by_num_atoms(tensor_map: TensorMap, num_atoms: torch.Tensor) -> TensorMap:
    """Takes the average values per atom of a ``TensorMap``.

    Since some quantities might already be per atom (e.g., atomic energies
    or position gradients), this function only divides a block (or gradient
    block) by the number of atoms if the block's samples do not contain
    the "atom" key. In practice, this guarantees the desired behavior for
    the majority of the cases, including energies, forces, and virials, where
    the energies and virials should be divided by the number of atoms, while
    the forces should not.

    :param tensor_map: The input tensor map.
    :param num_atoms: The number of atoms in each system.

    :return: A new tensor map with the values divided by the number of atoms.
    """

    blocks = []
    for block in tensor_map.blocks():
        if "atom" in block.samples.names:
            new_block = block
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
                if "atom" in gradient.samples.names:
                    new_gradient = gradient
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
