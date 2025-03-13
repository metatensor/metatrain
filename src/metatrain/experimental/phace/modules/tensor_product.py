from typing import List

import torch


def uncouple_features(features: List[torch.Tensor], U, padded_l_max: int):
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
    stacked_features = stacked_features.swapaxes(1, 2)
    uncoupled_features = stacked_features @ U.T
    uncoupled_features = uncoupled_features.swapaxes(1, 2)
    uncoupled_features = uncoupled_features.reshape(
        stacked_features.shape[0],
        padded_l_max + 1,
        padded_l_max + 1,
        uncoupled_features.shape[-1],
    )
    return uncoupled_features


def combine_uncoupled_features(
    uncoupled_features_1: List[torch.Tensor], uncoupled_features_2: List[torch.Tensor]
):
    new_uncoupled_features = []
    for u1, u2 in zip(uncoupled_features_1, uncoupled_features_2):
        new_uncoupled_features.append(torch.einsum("...ijf,...jkf->...ikf", u1, u2))
    return new_uncoupled_features


def couple_features(features, U, padded_l_max: int):
    # features is [..., padded_l_max+1, padded_l_max+1, n_features]
    # U is dense and [(padded_l_max+1)**2, (padded_l_max+1)**2]
    features = features.reshape(
        features.shape[0], (padded_l_max + 1) * (padded_l_max + 1), features.shape[-1]
    )
    features = features.swapaxes(1, 2)
    stacked_features = features @ U
    stacked_features = stacked_features.swapaxes(1, 2)
    split_sizes = [2 * l + 1 for l in range(padded_l_max + 1)]
    return torch.split(stacked_features, split_sizes, dim=1)
