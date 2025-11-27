from typing import List

import torch


def embed_centers_tensor_map(equivariants: List[torch.Tensor], center_embeddings: torch.Tensor):
    # multiplies arbitrary equivariant features by the provided center embeddings

    n_channels = center_embeddings.shape[-1]

    blocks: List[torch.Tensor] = []

    for block in equivariants:
        assert block.shape[-1] % n_channels == 0
        n_repeats = block.shape[-1] // n_channels
        new_block = block * center_embeddings.repeat(
            1, n_repeats
        ).unsqueeze(1)
        blocks.append(new_block)

    return blocks


def embed_centers(features: List[torch.Tensor], center_embeddings: torch.Tensor):
    # multiplies arbitrary equivariant features by the provided center embeddings

    n_channels = center_embeddings.shape[-1]

    new_features: List[torch.Tensor] = []
    for feature_tensor in features:
        assert feature_tensor.shape[-1] % n_channels == 0
        n_repeats = feature_tensor.shape[-1] // n_channels
        new_block_values = feature_tensor * center_embeddings.repeat(
            1, n_repeats
        ).unsqueeze(1)

        new_features.append((new_block_values))

    return new_features
