from typing import List

import torch
from metatensor.torch import TensorBlock, TensorMap


def embed_centers(equivariants: TensorMap, center_embeddings: torch.Tensor):
    # multiplies arbitrary equivariant features by the provided center embeddings

    n_channels = center_embeddings.shape[-1]

    keys: List[torch.Tensor] = []
    blocks: List[TensorBlock] = []

    for key, block in equivariants.items():
        assert block.values.shape[-1] % n_channels == 0
        n_repeats = block.values.shape[-1] // n_channels
        new_block_values = block.values * center_embeddings.repeat(
            1, n_repeats
        ).unsqueeze(1)
        keys.append(key.values)
        blocks.append(
            TensorBlock(
                values=new_block_values,
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
        )

    return TensorMap(
        keys=equivariants.keys,
        blocks=blocks,
    )
