from typing import List

import torch
from metatensor.torch.atomistic import System


class LongRangeFeaturizer(torch.nn.Module):
    def __init__(self, hypers):
        super(LongRangeFeaturizer, self).__init__()
        raise NotImplementedError
        # use hypers for long-range, register modules, etc

    def forward(
        systems: List[System],
        features: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_distances: torch.Tensor,
    ):
        raise NotImplementedError
        # implement long-range featurization here
