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
            U_dict[padded_l] = U

        U_dict_parity: Dict[str, torch.Tensor] = {}
        for padded_l in list(U_dict.keys()):
            U_dict_parity[f"{padded_l}_{1}"] = U_dict[padded_l].clone()
            # mask out odd l values
            for l in range(1, padded_l + 1, 2):
                U_dict_parity[f"{padded_l}_{1}"][:, l**2 : (l + 1) ** 2] = 0.0
            U_dict_parity[f"{padded_l}_{1}"] = U_dict_parity[
                f"{padded_l}_{1}"
            ].to_sparse_csr()
            # print(
            #     "Sparsity: ",
            #     U_dict_parity[f"{padded_l}_{1}"]._nnz()
            #     / U_dict_parity[f"{padded_l}_{1}"].numel(),
            # )
            U_dict_parity[f"{padded_l}_{-1}"] = U_dict[padded_l].clone()
            # mask out even l values
            for l in range(0, padded_l + 1, 2):
                U_dict_parity[f"{padded_l}_{-1}"][:, l**2 : (l + 1) ** 2] = 0.0
            U_dict_parity[f"{padded_l}_{-1}"] = U_dict_parity[
                f"{padded_l}_{-1}"
            ].to_sparse_csr()
            # print(
            #     "Sparsity: ",
            #     U_dict_parity[f"{padded_l}_{-1}"]._nnz()
            #     / U_dict_parity[f"{padded_l}_{-1}"].numel(),
            # )
        self.U_dict_parity = U_dict_parity

    def forward(self, features_1: List[torch.Tensor], features_2: List[torch.Tensor]):
        device = features_1[0].device
        if self.U_dict_parity[0].device != device:
            self.U_dict_parity = {
                key: U.to(device) for key, U in self.U_dict_parity.items()
            }
        dtype = features_2[0].dtype
        if self.U_dict_parity[0].dtype != dtype:
            self.U_dict_parity = {
                key: U.to(dtype) for key, U in self.U_dict_parity.items()
            }

        split_features_1 = split_up_features(features_1, self.k_max_l)
        split_features_2 = split_up_features(features_2, self.k_max_l)

        uncoupled_features_1: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for l in range(self.l_max + 1):
            uncoupled_features_1.append(
                uncouple_features(
                    split_features_1[l],
                    (
                        self.U_dict_parity[f"{self.padded_l_list[l]}_{1}"],
                        self.U_dict_parity[f"{self.padded_l_list[l]}_{-1}"],
                    ),
                    self.padded_l_list[l],
                )
            )

        uncoupled_features_2: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for l in range(self.l_max + 1):
            uncoupled_features_2.append(
                uncouple_features(
                    split_features_2[l],
                    (
                        self.U_dict_parity[f"{self.padded_l_list[l]}_{1}"],
                        self.U_dict_parity[f"{self.padded_l_list[l]}_{-1}"],
                    ),
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
                    (
                        self.U_dict_parity[f"{self.padded_l_list[l]}_{1}"],
                        self.U_dict_parity[f"{self.padded_l_list[l]}_{-1}"],
                    ),
                    self.padded_l_list[l],
                )[0]
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
    Us: Tuple[torch.Tensor, torch.Tensor],
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
    uncoupled_features_even = (
        Us[0]
        @ stacked_features.reshape(
            (padded_l_max + 1) * (padded_l_max + 1),
            stacked_features.shape[1] * stacked_features.shape[-1],
        )
    ).reshape(
        (padded_l_max + 1) * (padded_l_max + 1),
        stacked_features.shape[1],
        stacked_features.shape[-1],
    )
    uncoupled_features_even = uncoupled_features_even.swapaxes(0, 1)
    uncoupled_features_even = uncoupled_features_even.reshape(
        uncoupled_features_even.shape[0],
        padded_l_max + 1,
        padded_l_max + 1,
        uncoupled_features_even.shape[-1],
    )
    uncoupled_features_odd = (
        Us[1]
        @ stacked_features.reshape(
            (padded_l_max + 1) * (padded_l_max + 1),
            stacked_features.shape[1] * stacked_features.shape[-1],
        )
    ).reshape(
        (padded_l_max + 1) * (padded_l_max + 1),
        stacked_features.shape[1],
        stacked_features.shape[-1],
    )
    uncoupled_features_odd = uncoupled_features_odd.swapaxes(0, 1)
    uncoupled_features_odd = uncoupled_features_odd.reshape(
        uncoupled_features_odd.shape[0],
        padded_l_max + 1,
        padded_l_max + 1,
        uncoupled_features_odd.shape[-1],
    )
    return uncoupled_features_even, uncoupled_features_odd


def combine_uncoupled_features(
    uncoupled_features_1: List[Tuple[torch.Tensor, torch.Tensor]],
    uncoupled_features_2: List[Tuple[torch.Tensor, torch.Tensor]],
):
    new_uncoupled_features: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for u1, u2 in zip(uncoupled_features_1, uncoupled_features_2, strict=False):
        u_even_1, u_odd_1 = u1
        u_even_2, u_odd_2 = u2
        new_uncoupled_features_odd = torch.einsum(
            "...ijf,...jkf->...ikf", u_even_1, u_odd_2
        ) + torch.einsum("...ijf,...jkf->...ikf", u_odd_1, u_even_2)
        new_uncoupled_features_even = torch.einsum(
            "...ijf,...jkf->...ikf", u_even_1, u_even_2
        ) + torch.einsum("...ijf,...jkf->...ikf", u_odd_1, u_odd_2)
        new_uncoupled_features.append(
            (new_uncoupled_features_even, new_uncoupled_features_odd)
        )
    return new_uncoupled_features


def couple_features(
    features: Tuple[torch.Tensor, torch.Tensor],
    Us: Tuple[torch.Tensor, torch.Tensor],
    padded_l_max: int,
):
    # features is [..., padded_l_max+1, padded_l_max+1, n_features]
    # U is dense and [(padded_l_max+1)**2, (padded_l_max+1)**2]
    split_sizes = [2 * l + 1 for l in range(padded_l_max + 1)]
    features_even = features[0]

    features_even = features_even.reshape(
        features_even.shape[0],
        (padded_l_max + 1) * (padded_l_max + 1),
        features_even.shape[-1],
    )
    features_even = features_even.swapaxes(0, 1)
    stacked_features_from_even_uncoupled = (
        Us[0].T
        @ features_even.reshape(
            (padded_l_max + 1) * (padded_l_max + 1),
            features_even.shape[1] * features_even.shape[-1],
        )
    ).reshape(
        (padded_l_max + 1) * (padded_l_max + 1),
        features_even.shape[1],
        features_even.shape[-1],
    )
    stacked_features_from_even_uncoupled = (
        stacked_features_from_even_uncoupled.swapaxes(0, 1)
    )
    features_from_even_uncoupled = torch.split(
        stacked_features_from_even_uncoupled, split_sizes, dim=1
    )

    features_odd = features[1]
    features_odd = features_odd.reshape(
        features_odd.shape[0],
        (padded_l_max + 1) * (padded_l_max + 1),
        features_odd.shape[-1],
    )
    features_odd = features_odd.swapaxes(0, 1)
    stacked_features_from_odd_uncoupled = (
        Us[1].T
        @ features_odd.reshape(
            (padded_l_max + 1) * (padded_l_max + 1),
            features_odd.shape[1] * features_odd.shape[-1],
        )
    ).reshape(
        (padded_l_max + 1) * (padded_l_max + 1),
        features_odd.shape[1],
        features_odd.shape[-1],
    )
    stacked_features_from_odd_uncoupled = stacked_features_from_odd_uncoupled.swapaxes(
        0, 1
    )
    features_from_odd_uncoupled = torch.split(
        stacked_features_from_odd_uncoupled, split_sizes, dim=1
    )

    even_coupled_features = []
    odd_coupled_features = []
    for l in range(padded_l_max + 1):
        if l % 2 == 0:
            even_coupled_features.append(features_from_even_uncoupled[l])
            odd_coupled_features.append(features_from_odd_uncoupled[l])
        else:
            even_coupled_features.append(features_from_odd_uncoupled[l])
            odd_coupled_features.append(features_from_even_uncoupled[l])
    return even_coupled_features, odd_coupled_features
