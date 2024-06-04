from typing import List

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


class LinearMap(torch.nn.Module):
    # TODO: replace with the one from metatensor-learn once released
    def __init__(self, n_inputs: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(n_inputs, 1)

    def forward(self, features: TensorMap) -> TensorMap:

        # assume invariant
        block = features.block({"o3_lambda": 0, "o3_sigma": 1})

        output_values = self.linear(block.values).squeeze(1)
        new_block = TensorBlock(
            values=output_values,
            samples=block.samples,
            components=[],
            properties=Labels(
                names=["energy"],
                values=torch.zeros(
                    (1, 1), dtype=torch.int32, device=block.values.device
                ),
            ),
        )

        return TensorMap(
            keys=Labels(
                names=["_"],
                values=torch.zeros(
                    1, 1, dtype=torch.int32, device=new_block.values.device
                ),
            ),
            blocks=[new_block],
        )
