from typing import List

import torch

from .layers import LinearList as Linear
from .tensor_product import combine_uncoupled_features


class CGIterator(torch.nn.Module):
    # A high-level CG iterator, doing multiple iterations
    def __init__(
        self,
        k_max_l_max: List[int],
        number_of_iterations,
    ):
        super().__init__()
        self.k_max_l_max = k_max_l_max
        self.number_of_iterations = number_of_iterations

        # equivariant linear mixers (to be used at the beginning)
        mixers = []
        for _ in range(self.number_of_iterations + 1):
            mixers.append(Linear(k_max_l_max))
        self.mixers = torch.nn.ModuleList(mixers)

        # CG iterations
        cg_iterations = []
        for n_iteration in range(self.number_of_iterations):
            cg_iterations.append(CGIteration(self.k_max_l_max))
        self.cg_iterations = torch.nn.ModuleList(cg_iterations)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:

        density = features
        mixed_densities = [mixer(density) for mixer in self.mixers]

        starting_density = mixed_densities[0]
        density_index = 1
        current_density = starting_density
        for iterator in self.cg_iterations:
            current_density = iterator(current_density, mixed_densities[density_index])
            density_index += 1

        return current_density


class CGIteration(torch.nn.Module):
    # A single Clebsch-Gordan iteration, including:
    # - tensor product
    # - linear transformation
    # - skip connection
    def __init__(
        self,
        k_max_l_max: List[int],
    ):
        super().__init__()
        self.linear = Linear(k_max_l_max)

    def forward(
        self, features_1: List[torch.Tensor], features_2: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        features_out = combine_uncoupled_features(features_1, features_2)
        features_out = self.linear(features_out)
        features_out = [f1 + fo for f1, fo in zip(features_1, features_out)]
        return features_out
