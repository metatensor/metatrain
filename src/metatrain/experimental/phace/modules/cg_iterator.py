from typing import List

import torch

from .layers import LinearList as Linear
from .tensor_product import TensorProduct


class CGIterator(torch.nn.Module):
    # A high-level CG iterator, doing multiple iterations
    def __init__(
        self,
        tensor_product: TensorProduct,
        number_of_iterations,
    ):
        super().__init__()
        self.k_max_l_max = tensor_product.k_max_l
        self.number_of_iterations = number_of_iterations

        # equivariant linear mixers (to be used at the beginning)
        mixers = []
        for _ in range(self.number_of_iterations + 1):
            mixers.append(Linear(tensor_product.k_max_l))
        self.mixers = torch.nn.ModuleList(mixers)

        # CG iterations
        cg_iterations = []
        for n_iteration in range(self.number_of_iterations):
            cg_iterations.append(CGIteration(tensor_product))
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
        tensor_product: TensorProduct,
    ):
        super().__init__()
        self.tensor_product = tensor_product
        self.linear = Linear(tensor_product.k_max_l)

    def forward(
        self, features_1: List[torch.Tensor], features_2: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        features_out = self.tensor_product(features_1, features_2)
        features_out = self.linear(features_out)
        features_out = [
            f1 + fo for f1, fo in zip(features_1, features_out, strict=False)
        ]
        return features_out
