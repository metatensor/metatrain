from typing import Dict, List

import torch


class Linear(torch.nn.Module):
    """NTK-style linear layer (NTK = neural tangent kernel)."""

    def __init__(self, n_feat_in, n_feat_out):
        super().__init__()
        self.linear_layer = torch.nn.Linear(n_feat_in, n_feat_out, bias=False)
        self.linear_layer.weight.data.normal_(0.0, 1.0)
        self.n_feat_in = n_feat_in if n_feat_in > 0 else 1

    def forward(self, x):
        return self.linear_layer(x) * self.n_feat_in ** (-0.5)


class LinearList(torch.nn.Module):
    """List of linear layers for a list of features in the compact (uncoupled) basis."""

    def __init__(
        self,
        k_max_l: List[int],
        expansion_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.k_max_l = k_max_l
        self.l_max = len(k_max_l) - 1
        self.padded_l_list = [2 * ((l + 1) // 2) for l in range(self.l_max + 1)]  # noqa: E741
        l_max = len(k_max_l) - 1
        self.linears = []
        for l in range(l_max, -1, -1):  # noqa: E741
            lower_bound = k_max_l[l + 1] if l < l_max else 0
            upper_bound = k_max_l[l]
            dimension = upper_bound - lower_bound
            self.linears.append(Linear(dimension, int(dimension * expansion_ratio)))
        self.linears = torch.nn.ModuleList(self.linears[::-1])

    def forward(
        self, features_list: List[torch.Tensor], U_dict: Dict[int, torch.Tensor]
    ) -> List[torch.Tensor]:
        new_features_list: List[torch.Tensor] = []
        for i, linear in enumerate(self.linears):
            current_features = features_list[i]
            new_features = linear(current_features)
            new_features_list.append(new_features)

        return new_features_list


class BlockRMSNorm(torch.nn.Module):
    """RMSNorm applied to a specific l block in the uncoupled basis.

    Equivariance is preserved since the norm is computed over the feature dimension
    and the block structure is preserved.
    """

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d))

    def forward(self, x):  # e.g., x: [..., 3, 3, 128]
        rms_inv = x.pow(2).mean(dim=(-3, -2, -1), keepdim=True).add(self.eps).rsqrt()
        x = x * rms_inv
        return x * self.weight


class EquivariantRMSNorm(torch.nn.Module):
    """RMSNorm applied to all features in a ragged compact (uncoupled) basis.

    Inputs will look like ``[[..., 1, 1, 128], [..., 3, 3, 0], [..., 3, 3, 128]]``
    with default hyperparameters.
    """

    def __init__(self, k_max_l: List[int]) -> None:
        super().__init__()
        self.k_max_l = k_max_l
        self.l_max = len(k_max_l) - 1

        l_max = len(k_max_l) - 1
        self.rmsnorms = []
        for l in range(l_max, -1, -1):  # noqa: E741
            lower_bound = k_max_l[l + 1] if l < l_max else 0
            upper_bound = k_max_l[l]
            dimension = upper_bound - lower_bound
            self.rmsnorms.append(BlockRMSNorm(dimension))
        self.rmsnorms = torch.nn.ModuleList(self.rmsnorms[::-1])

    def forward(self, features_list: List[torch.Tensor]) -> List[torch.Tensor]:
        new_features_list: List[torch.Tensor] = []
        for i, rmsnorm in enumerate(self.rmsnorms):
            new_features = rmsnorm(features_list[i])
            new_features_list.append(new_features)
        return new_features_list
