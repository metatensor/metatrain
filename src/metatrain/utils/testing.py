# This file contains some example TensorMap layouts that can be
# used for testing purposes.

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


block = TensorBlock(
    # float64: otherwise metatensor can't serialize
    values=torch.empty(0, 1, dtype=torch.float64),
    samples=Labels(
        names=["system"],
        values=torch.empty((0, 1), dtype=torch.int32),
    ),
    components=[],
    properties=Labels.range("energy", 1),
)
energy_layout = TensorMap(
    keys=Labels.single(),
    blocks=[block],
)

block_with_position_gradients = block.copy()
position_gradient_block = TensorBlock(
    # float64: otherwise metatensor can't serialize
    values=torch.empty(0, 3, 1, dtype=torch.float64),
    samples=Labels(
        names=["sample", "atom"],
        values=torch.empty((0, 2), dtype=torch.int32),
    ),
    components=[
        Labels(
            names=["xyz"],
            values=torch.arange(3, dtype=torch.int32).reshape(-1, 1),
        ),
    ],
    properties=Labels.range("energy", 1),
)
block_with_position_gradients.add_gradient("positions", position_gradient_block)
energy_force_layout = TensorMap(
    keys=Labels.single(),
    blocks=[block_with_position_gradients],
)

block_with_position_and_strain_gradients = block_with_position_gradients.copy()
strain_gradient_block = TensorBlock(
    # float64: otherwise metatensor can't serialize
    values=torch.empty(0, 3, 3, 1, dtype=torch.float64),
    samples=Labels(
        names=["sample", "atom"],
        values=torch.empty((0, 2), dtype=torch.int32),
    ),
    components=[
        Labels(
            names=["xyz_1"],
            values=torch.arange(3, dtype=torch.int32).reshape(-1, 1),
        ),
        Labels(
            names=["xyz_2"],
            values=torch.arange(3, dtype=torch.int32).reshape(-1, 1),
        ),
    ],
    properties=Labels.range("energy", 1),
)
block_with_position_and_strain_gradients.add_gradient("strain", strain_gradient_block)
energy_force_stress_layout = TensorMap(
    keys=Labels.single(),
    blocks=[block_with_position_and_strain_gradients],
)

block_with_strain_gradients = block.copy()
block_with_strain_gradients.add_gradient("strain", strain_gradient_block)
energy_stress_layout = TensorMap(
    keys=Labels.single(),
    blocks=[block_with_strain_gradients],
)
