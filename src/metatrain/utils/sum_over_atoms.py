import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


@torch.jit.script
def sum_over_atoms(tensor_map: TensorMap):
    block = tensor_map.block()
    n_systems = int(block.samples.column("system").max() + 1)
    new_tensor = torch.zeros(
        (n_systems, block.values.shape[-1]),
        device=tensor_map.device,
        dtype=tensor_map.dtype,
    )
    new_tensor.index_add_(0, block.samples.column("system"), block.values)
    return TensorMap(
        keys=tensor_map.keys,
        blocks=[
            TensorBlock(
                values=new_tensor,
                samples=Labels(
                    names=["system"],
                    values=torch.arange(
                        n_systems, device=tensor_map.device, dtype=torch.int
                    ).reshape(-1, 1),
                ),
                components=block.components,
                properties=block.properties,
            )
        ],
    )
