from typing import List

import torch

from .layers import Linear


class MLPRadialBasis(torch.nn.Module):
    """
    A module that applies a multi-layer perceptron to pre-computed radial basis
    functions, separately for each l.
    """

    def __init__(self, n_max_l, k_max_l, depth=3, width_factor=4) -> None:
        super().__init__()
        l_max = len(n_max_l) - 1
        if depth <= 1:
            raise ValueError("Radial MLP depth must be at least 2")

        self.radial_mlps = torch.nn.ModuleDict(
            {
                str(l): torch.nn.Sequential(
                    # input layer
                    Linear(n_max_l[l], width_factor * k_max_l[l]),
                    torch.nn.SiLU(),
                    # hidden layers
                    *[
                        layer
                        for _ in range(depth - 1)
                        for layer in (
                            Linear(
                                width_factor * k_max_l[l], width_factor * k_max_l[l]
                            ),
                            torch.nn.SiLU(),
                        )
                    ],
                    # output layer
                    Linear(width_factor * k_max_l[l], k_max_l[l]),
                )
                for l in range(l_max + 1)  # noqa: E741
            }
        )

    def forward(self, radial_basis: List[torch.Tensor]) -> List[torch.Tensor]:
        radial_basis_after_mlp = []
        for l_string, radial_mlp_l in self.radial_mlps.items():
            l = int(l_string)  # noqa: E741
            radial_basis_after_mlp.append(radial_mlp_l(radial_basis[l]))
        radial_basis = radial_basis_after_mlp
        return radial_basis
