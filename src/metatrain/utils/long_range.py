import torch


class LongRangeFeaturizer(torch.nn.Module):
    def __init__(self, hypers):
        super(LongRangeFeaturizer, self).__init__()
        raise NotImplementedError
        # use hypers for long-range, register modules, etc

    def forward(systems, features, neighbor_indices, neighbor_distances):
        raise NotImplementedError
        # implement long-range featurization here
