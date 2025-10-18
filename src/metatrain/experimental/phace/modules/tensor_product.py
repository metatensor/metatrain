from typing import Dict, List, Tuple

import numpy as np
import torch

from .cg import get_cg_coefficients


class TensorProduct(torch.nn.Module):
    def __init__(self, k_max_l: List[int]):
        super().__init__()
        self.k_max_l = k_max_l
        self.l_max = len(k_max_l) - 1

        cg_calculator = get_cg_coefficients(2 * ((self.l_max + 1) // 2))
        self.padded_l_list = [2 * ((l + 1) // 2) for l in range(self.l_max + 1)]
        U_dict = {}
        for padded_l in np.unique(self.padded_l_list):
            cg_tensors = [
                cg_calculator._cgs[(padded_l // 2, padded_l // 2, L)]
                for L in range(padded_l + 1)
            ]
            U = torch.concatenate(
                [cg_tensor for cg_tensor in cg_tensors], dim=2
            ).reshape((padded_l + 1) ** 2, (padded_l + 1) ** 2)
            assert torch.allclose(
                U @ U.T, torch.eye((padded_l + 1) ** 2, dtype=U.dtype)
            )
            assert torch.allclose(
                U.T @ U, torch.eye((padded_l + 1) ** 2, dtype=U.dtype)
            )
            U_dict[int(padded_l)] = U
        self.U_dict = U_dict

    def forward(self, features_1: List[torch.Tensor], features_2: List[torch.Tensor]):
        device = features_1[0].device
        if self.U_dict[0].device != device:
            self.U_dict = {key: U.to(device) for key, U in self.U_dict.items()}
        dtype = features_2[0].dtype
        if self.U_dict[0].dtype != dtype:
            self.U_dict = {key: U.to(dtype) for key, U in self.U_dict.items()}

        split_features_1 = split_up_features(features_1, self.k_max_l)
        split_features_2 = split_up_features(features_2, self.k_max_l)

        uncoupled_features_1: List[torch.Tensor] = []
        for l in range(self.l_max + 1):
            uncoupled_features_1.append(
                uncouple_features(
                    split_features_1[l],
                    self.U_dict[self.padded_l_list[l]],
                    self.padded_l_list[l],
                )
            )

        uncoupled_features_2 = []
        for l in range(self.l_max + 1):
            uncoupled_features_2.append(
                uncouple_features(
                    split_features_2[l],
                    self.U_dict[self.padded_l_list[l]],
                    self.padded_l_list[l],
                )
            )

        combined_features = combine_uncoupled_features(
            uncoupled_features_1, uncoupled_features_2
        )

        coupled_features: List[List[torch.Tensor]] = []
        for l in range(self.l_max + 1):
            coupled_features.append(
                couple_features(
                    combined_features[l],
                    self.U_dict[self.padded_l_list[l]],
                    self.padded_l_list[l],
                )
            )

        concatenated_coupled_features = []
        for l in range(self.l_max + 1):
            concatenated_coupled_features.append(
                torch.concatenate(
                    [coupled_features[lp][l] for lp in range(l, self.l_max + 1)], dim=-1
                )
            )

        return concatenated_coupled_features


def split_up_features(features: List[torch.Tensor], k_max_l: List[int]):
    l_max = len(k_max_l) - 1
    split_features: List[List[torch.Tensor]] = []
    for l in range(l_max, -1, -1):
        lower_bound = k_max_l[l + 1] if l < l_max else 0
        upper_bound = k_max_l[l]
        split_features = [
            [features[lp][:, :, lower_bound:upper_bound] for lp in range(l + 1)]
        ] + split_features
    return split_features


def uncouple_features(
    features: List[torch.Tensor],
    U: torch.Tensor,
    padded_l_max: int,
):
    # features is a list of [..., 2*l+1, n_features] for l = 0, 1, ..., padded_l_max
    # U is dense and [(padded_l_max+1)**2, (padded_l_max+1)**2]
    if len(features) < padded_l_max + 1:
        features.append(
            torch.zeros(
                (features[0].shape[0], 2 * padded_l_max + 1, features[0].shape[2]),
                dtype=features[0].dtype,
                device=features[0].device,
            )
        )
    stacked_features = torch.cat(features, dim=1)
    stacked_features = stacked_features.swapaxes(0, 1)
    uncoupled_features = (
        U @ stacked_features.reshape(
            (padded_l_max + 1) * (padded_l_max + 1),
            stacked_features.shape[1] * stacked_features.shape[-1],
        )
    ).reshape(
        (padded_l_max + 1) * (padded_l_max + 1),
        stacked_features.shape[1],
        stacked_features.shape[-1],
    )
    uncoupled_features = uncoupled_features.swapaxes(0, 1)
    uncoupled_features = uncoupled_features.reshape(
        uncoupled_features.shape[0],
        padded_l_max + 1,
        padded_l_max + 1,
        uncoupled_features.shape[-1],
    )
    return uncoupled_features


def combine_uncoupled_features(
    uncoupled_features_1: List[torch.Tensor],
    uncoupled_features_2: List[torch.Tensor],
):
    new_uncoupled_features = []
    for u1, u2 in zip(uncoupled_features_1, uncoupled_features_2, strict=True):
        new_uncoupled_features.append(torch.einsum("...ijf,...jkf->...ikf", u1, u2))
    return new_uncoupled_features


def couple_features(
    features: torch.Tensor,
    U: torch.Tensor,
    padded_l_max: int,
):
    # features is [..., padded_l_max+1, padded_l_max+1, n_features]
    # U is dense and [(padded_l_max+1)**2, (padded_l_max+1)**2]
    split_sizes = [2 * l + 1 for l in range(padded_l_max + 1)]

    features = features.reshape(
        features.shape[0],
        (padded_l_max + 1) * (padded_l_max + 1),
        features.shape[-1],
    )
    features = features.swapaxes(0, 1)
    features = (
        U.T @ features.reshape(
            (padded_l_max + 1) * (padded_l_max + 1),
            features.shape[1] * features.shape[-1],
        )
    ).reshape(
        (padded_l_max + 1) * (padded_l_max + 1),
        features.shape[1],
        features.shape[-1],
    )
    stacked_features = (
        features.swapaxes(0, 1)
    )
    features_coupled = torch.split(
        stacked_features, split_sizes, dim=1
    )

    coupled_features = []
    for l in range(padded_l_max + 1):
        coupled_features.append(features_coupled[l])
    return coupled_features
