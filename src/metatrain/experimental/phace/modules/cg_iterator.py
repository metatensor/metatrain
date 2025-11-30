from typing import List

import torch

from .layers import LinearList as Linear
from .tensor_product import tensor_product


class CGIterator(torch.nn.Module):
    # A high-level CG iterator, doing multiple iterations
    def __init__(
        self,
        k_max_l,
        number_of_iterations,
        spherical_linear_layers
    ):
        super().__init__()
        self.number_of_iterations = number_of_iterations

        # equivariant linear mixers (to be used at the beginning)
        mixers = []
        for _ in range(self.number_of_iterations + 1):
            mixers.append(Linear(k_max_l, spherical_linear_layers))
        self.mixers = torch.nn.ModuleList(mixers)

        # CG iterations
        cg_iterations = []
        for _ in range(self.number_of_iterations):
            cg_iterations.append(CGIteration(k_max_l, spherical_linear_layers))
        self.cg_iterations = torch.nn.ModuleList(cg_iterations)

    def forward(self, features: List[torch.Tensor], U_dict) -> List[torch.Tensor]:
        density = features
        mixed_densities = [mixer(density, U_dict) for mixer in self.mixers]

        starting_density = mixed_densities[0]
        density_index = 1
        current_density = starting_density
        for iterator in self.cg_iterations:
            current_density = iterator(current_density, mixed_densities[density_index], U_dict)
            density_index += 1

        return current_density


class CGIteration(torch.nn.Module):
    # A single Clebsch-Gordan iteration, including:
    # - tensor product
    # - linear transformation
    # - skip connection
    def __init__(
        self,
        k_max_l,
        spherical_linear_layers
    ):
        super().__init__()
        self.linear = Linear(k_max_l, spherical_linear_layers)

    def forward(
        self, features_1: List[torch.Tensor], features_2: List[torch.Tensor], U_dict
    ) -> List[torch.Tensor]:
        features_out = tensor_product(features_1, features_2)
        features_out = self.linear(features_out, U_dict)
        features_out = [
            f1 + fo for f1, fo in zip(features_1, features_out, strict=True)
        ]
        return features_out
