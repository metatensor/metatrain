from typing import List, Tuple

import torch
from metatensor.torch import TensorBlock, TensorMap


def embed_centers_tensor_map(equivariants: TensorMap, center_embeddings: torch.Tensor):
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


def embed_centers(features: List[Tuple[torch.Tensor, torch.Tensor]], center_embeddings: torch.Tensor):
    # multiplies arbitrary equivariant features by the provided center embeddings

    n_channels = center_embeddings.shape[-1]

    new_features: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for feature_tensor_even, feature_tensor_odd in features:
        assert feature_tensor_even.shape[-1] % n_channels == 0
        n_repeats = feature_tensor_even.shape[-1] // n_channels
        new_block_values_even = feature_tensor_even * center_embeddings.repeat(
            1, n_repeats
        ).unsqueeze(1).unsqueeze(2)

        assert feature_tensor_odd.shape[-1] % n_channels == 0
        n_repeats = feature_tensor_odd.shape[-1] // n_channels
        new_block_values_odd = feature_tensor_odd * center_embeddings.repeat(
            1, n_repeats
        ).unsqueeze(1).unsqueeze(2)

        new_features.append((new_block_values_even, new_block_values_odd))

    return new_features
