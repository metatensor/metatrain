import torch
from metatensor.torch import TensorBlock


def average_block_by_num_atoms(
    block: TensorBlock, num_atoms: torch.Tensor
) -> TensorBlock:
    """Takes the average values per atom of a `TensorBlock`."""

    new_values = block.values / num_atoms
    new_block = TensorBlock(
        values=new_values,
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )
    for param, gradient in block.gradients():
        new_block.add_gradient(param, gradient)

    return new_block
