from typing import List

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from . import torch_jit_script_unless_coverage


@torch_jit_script_unless_coverage
def sum_over_atoms(tensor_map: TensorMap):
    """
    A faster version of ``metatensor.torch.sum_over_samples``, specialized for
    summing over atoms in graph-like TensorMaps.

    :param tensor_map: The TensorMap to sum over.
    :return: A new TensorMap with the same keys, but with the samples summed
        over the atoms.
    """
    new_blocks: List[TensorBlock] = []
    for block in tensor_map.blocks():
        device = block.values.device
        dtype = block.values.dtype
        n_systems = int(block.samples.column("system").max() + 1)
        new_tensor = torch.zeros(
            [n_systems] + list(block.values.shape[1:]), device=device, dtype=dtype
        )
        new_tensor.index_add_(0, block.samples.column("system"), block.values)
        new_block = TensorBlock(
            values=new_tensor,
            samples=Labels(
                names=["system"],
                values=torch.arange(
                    n_systems, device=device, dtype=torch.int32
                ).reshape(-1, 1),
                assume_unique=True,
            ),
            components=block.components,
            properties=block.properties,
        )
        new_blocks.append(new_block)
    return TensorMap(
        keys=tensor_map.keys,
        blocks=new_blocks,
    )
