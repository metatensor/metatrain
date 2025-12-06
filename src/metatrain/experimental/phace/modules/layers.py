from typing import Dict, List

import torch

from .tensor_product import couple_features_all, uncouple_features_all


class Linear(torch.nn.Module):
    # NTK-style linear layer (neural tangent kernel)

    def __init__(self, n_feat_in, n_feat_out):
        super().__init__()
        self.linear_layer = torch.nn.Linear(n_feat_in, n_feat_out, bias=False)
        self.linear_layer.weight.data.normal_(0.0, 1.0)
        self.n_feat_in = n_feat_in if n_feat_in > 0 else 1

    def forward(self, x):
        return self.linear_layer(x) * self.n_feat_in ** (-0.5)


class LinearList(torch.nn.Module):
    # list of linear layers for equivariant features, either in the spherical basis
    # (spherical_linear_layers=True) or in the coupled (TP) basis

    def __init__(self, k_max_l: List[int], spherical_linear_layers) -> None:
        super().__init__()
        self.spherical_linear_layers = spherical_linear_layers
        self.k_max_l = k_max_l
        self.l_max = len(k_max_l) - 1
        self.padded_l_list = [2 * ((l + 1) // 2) for l in range(self.l_max + 1)]  # noqa: E741
        if spherical_linear_layers:
            self.linears = torch.nn.ModuleList(
                [Linear(k_max, k_max) for k_max in k_max_l]
            )
        else:
            l_max = len(k_max_l) - 1
            self.linears = []
            for l in range(l_max, -1, -1):  # noqa: E741
                lower_bound = k_max_l[l + 1] if l < l_max else 0
                upper_bound = k_max_l[l]
                dimension = upper_bound - lower_bound
                self.linears.append(Linear(dimension, dimension))
            self.linears = torch.nn.ModuleList(self.linears[::-1])

    def forward(
        self, features_list: List[torch.Tensor], U_dict: Dict[int, torch.Tensor]
    ) -> List[torch.Tensor]:
        if self.spherical_linear_layers:
            features_list = couple_features_all(
                features_list, U_dict, self.l_max, self.padded_l_list
            )

        new_features_list: List[torch.Tensor] = []
        for i, linear in enumerate(self.linears):
            current_features = features_list[i]
            new_features = linear(current_features)
            new_features_list.append(new_features)

        if self.spherical_linear_layers:
            new_features_list = uncouple_features_all(
                new_features_list, self.k_max_l, U_dict, self.l_max, self.padded_l_list
            )

        return new_features_list
