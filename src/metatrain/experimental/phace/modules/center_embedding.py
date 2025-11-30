from typing import List
import torch


def embed_centers(features: List[torch.Tensor], center_embeddings: torch.Tensor):
    # multiplies arbitrary equivariant features by the provided center embeddings
    n_channels = center_embeddings.shape[-1]
    new_features: List[torch.Tensor] = []
    for feature_tensor in features:
        assert feature_tensor.shape[-1] % n_channels == 0
        n_repeats = feature_tensor.shape[-1] // n_channels
        center_embeddings_broadcast = center_embeddings.repeat(1, n_repeats)
        for _ in range(len(feature_tensor.shape) - len(center_embeddings.shape)):
            center_embeddings_broadcast = center_embeddings_broadcast.unsqueeze(1)
        new_block_values = feature_tensor * center_embeddings_broadcast
        new_features.append((new_block_values))
    return new_features
