import torch
from metatensor.torch import TensorBlock, TensorMap


def divide_by_num_atoms(tensor_map: TensorMap, num_atoms: torch.Tensor) -> TensorMap:
    """Takes the average values per atom of a ``TensorMap``.

    Since some quantities might already be per atom (e.g., atomic energies
    or position gradients), this function only divides a block (or gradient
    block) by the number of atoms if the block's samples do not contain
    the "atom" key. In practice, this guarantees the desired behavior for
    the majority of the cases, including energies, forces, and virials, where
    the energies and virials should be divided by the number of atoms, while
    the forces should not.
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
