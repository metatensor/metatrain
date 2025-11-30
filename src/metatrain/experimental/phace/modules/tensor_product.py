from typing import List
import torch


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
        U
        @ stacked_features.reshape(
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


def tensor_product(
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
        U.T
        @ features.reshape(
            (padded_l_max + 1) * (padded_l_max + 1),
            features.shape[1] * features.shape[-1],
        )
    ).reshape(
        (padded_l_max + 1) * (padded_l_max + 1),
        features.shape[1],
        features.shape[-1],
    )
    stacked_features = features.swapaxes(0, 1)
    features_coupled = [t.contiguous() for t in torch.split(stacked_features, split_sizes, dim=1)]

    coupled_features = []
    for l in range(padded_l_max + 1):
        coupled_features.append(features_coupled[l])
    return coupled_features
