from typing import List

import torch

from .layers import Linear


class MLPRadialBasis(torch.nn.Module):
    def __init__(self, n_max_l, k_max_l) -> None:
        super().__init__()

        l_max = len(n_max_l) - 1
        self.radial_mlps = torch.nn.ModuleDict(
            {
                str(l): torch.nn.Sequential(
                    Linear(n_max_l[l], 4 * k_max_l[l]),
                    torch.nn.SiLU(),
                    Linear(
                        4 * k_max_l[l],
                        4 * k_max_l[l],
                    ),
                    torch.nn.SiLU(),
                    Linear(
                        4 * k_max_l[l],
                        4 * k_max_l[l],
                    ),
                    torch.nn.SiLU(),
                    Linear(
                        4 * k_max_l[l],
                        k_max_l[l],
                    ),
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
