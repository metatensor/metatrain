
import torch

from .layers import Linear
from typing import List


class MLPRadialBasis(torch.nn.Module):
    def __init__(self, n_max_l, num_element_channels) -> None:
        super().__init__()

        self.n_max_l = list(n_max_l)
        self.l_max = len(self.n_max_l) - 1
        self.n_channels = num_element_channels

        self.apply_mlp = False
        self.radial_mlps = torch.nn.ModuleDict(
            {
                str(l): torch.nn.Sequential(
                    Linear(self.n_max_l[l], 4 * self.n_max_l[l] * self.n_channels),
                    torch.nn.SiLU(),
                    Linear(
                        4 * self.n_max_l[l] * self.n_channels,
                        4 * self.n_max_l[l] * self.n_channels,
                    ),
                    torch.nn.SiLU(),
                    Linear(
                        4 * self.n_max_l[l] * self.n_channels,
                        4 * self.n_max_l[l] * self.n_channels,
                    ),
                    torch.nn.SiLU(),
                    Linear(
                        4 * self.n_max_l[l] * self.n_channels,
                        self.n_max_l[l] * self.n_channels,
                        # 128
                    ),
                )
                for l in range(self.l_max + 1)  # noqa: E741
            }
        )

    def forward(self, radial_basis: List[torch.Tensor]) -> List[torch.Tensor]:

        radial_basis_after_mlp = []
        for l_string, radial_mlp_l in self.radial_mlps.items():
            l = int(l_string)  # noqa: E741
            radial_basis_after_mlp.append(radial_mlp_l(radial_basis[l]))
        radial_basis = radial_basis_after_mlp

        return radial_basis
