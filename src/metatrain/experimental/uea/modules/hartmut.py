import torch

from .cg import get_cg_coefficients


class SphericalToHartmut(torch.nn.Module):
    def __init__(self, l_max):
        super().__init__()

        if l_max % 2 != 0:
            raise ValueError("l_max must be even for HartmutConverter")

        self.l_max = l_max

        cg_calculator = get_cg_coefficients(2 * ((self.l_max + 1) // 2))
        # padded_l_max = 2 * ((self.l_max + 1) // 2)
        # self.padded_l_max = padded_l_max

        cg_tensors = [
            cg_calculator._cgs[(l_max // 2, l_max // 2, L)]
            for L in range(l_max + 1)
        ]
        U = torch.concatenate(
            [cg_tensor for cg_tensor in cg_tensors], dim=2
        ).reshape((l_max + 1) ** 2, (l_max + 1) ** 2)
        assert torch.allclose(
            U @ U.T, torch.eye((l_max + 1) ** 2, dtype=U.dtype)
        )
        assert torch.allclose(
            U.T @ U, torch.eye((l_max + 1) ** 2, dtype=U.dtype)
        )
        self.U = U

    def forward(self, spherical_features: torch.Tensor) -> torch.Tensor:
        if self.U.device != spherical_features.device:
            self.U = self.U.to(spherical_features.device)
        if self.U.dtype != spherical_features.dtype:
            self.U = self.U.to(spherical_features.dtype)

        hartmut_features = spherical_features @ self.U.T
        hartmut_features = hartmut_features.reshape(
            hartmut_features.shape[:-1] + (self.l_max + 1, self.l_max + 1)
        )

        return hartmut_features

    def back_to_spherical(self, hartmut_features: torch.Tensor) -> torch.Tensor:
        if self.U.device != hartmut_features.device:
            self.U = self.U.to(hartmut_features.device)
        if self.U.dtype != hartmut_features.dtype:
            self.U = self.U.to(hartmut_features.dtype)

        spherical_features = hartmut_features.reshape(
            hartmut_features.shape[:-2] + ((self.l_max + 1) * (self.l_max + 1),)
        ) @ self.U

        return spherical_features
