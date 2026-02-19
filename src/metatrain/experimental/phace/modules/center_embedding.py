from typing import List

import torch


class CenterEmbedder(torch.nn.Module):
    def __init__(self, num_atomic_types, num_channels_per_tensor):
        super().__init__()
        self.embedders = []
        for num_channels in num_channels_per_tensor:
            self.embedders.append(torch.nn.Embedding(num_atomic_types, num_channels))
        self.embedders = torch.nn.ModuleList(self.embedders)

    def forward(self, features: List[torch.Tensor], atomic_types: torch.Tensor):
        new_features = []
        for i, embedder in enumerate(self.embedders):
            embeddings = embedder(atomic_types)
            for _ in range(features[i].dim() - 2):
                embeddings = embeddings.unsqueeze(1)
            if features[i].shape[-1] != 0:
                new_features.append(features[i] * embeddings)
            else:
                new_features.append(features[i])
        return new_features
