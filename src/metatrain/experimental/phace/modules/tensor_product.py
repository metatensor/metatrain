from typing import List, Tuple

import torch


def uncouple_features(features: List[torch.Tensor], Us: Tuple[torch.Tensor, torch.Tensor], padded_l_max: int):
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
    uncoupled_features_even = (Us[0] @ stacked_features.reshape(
        (padded_l_max + 1) * (padded_l_max + 1), stacked_features.shape[1] * stacked_features.shape[-1]
    )).reshape(
        (padded_l_max + 1) * (padded_l_max + 1), stacked_features.shape[1], stacked_features.shape[-1]
    )
    uncoupled_features_even = uncoupled_features_even.swapaxes(0, 1)
    uncoupled_features_even = uncoupled_features_even.reshape(
        uncoupled_features_even.shape[0],
        padded_l_max + 1,
        padded_l_max + 1,
        uncoupled_features_even.shape[-1],
    )
    uncoupled_features_odd = (Us[1] @ stacked_features.reshape(
        (padded_l_max + 1) * (padded_l_max + 1), stacked_features.shape[1] * stacked_features.shape[-1]
    )).reshape(
        (padded_l_max + 1) * (padded_l_max + 1), stacked_features.shape[1], stacked_features.shape[-1]
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
    uncoupled_features_1: List[Tuple[torch.Tensor, torch.Tensor]], uncoupled_features_2: List[Tuple[torch.Tensor, torch.Tensor]]
):
    new_uncoupled_features: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for u1, u2 in zip(uncoupled_features_1, uncoupled_features_2):
        u_even_1, u_odd_1 = u1
        u_even_2, u_odd_2 = u2
        new_uncoupled_features_odd = torch.einsum("...ijf,...jkf->...ikf", u_even_1, u_odd_2) + torch.einsum("...ijf,...jkf->...ikf", u_odd_1, u_even_2)
        new_uncoupled_features_even = torch.einsum("...ijf,...jkf->...ikf", u_even_1, u_even_2) + torch.einsum("...ijf,...jkf->...ikf", u_odd_1, u_odd_2)
        new_uncoupled_features.append(
            (new_uncoupled_features_even, new_uncoupled_features_odd)
        )
    return new_uncoupled_features


def couple_features(features: Tuple[torch.Tensor, torch.Tensor], Us: Tuple[torch.Tensor, torch.Tensor], padded_l_max: int):
    # features is [..., padded_l_max+1, padded_l_max+1, n_features]
    # U is dense and [(padded_l_max+1)**2, (padded_l_max+1)**2]
    split_sizes = [2 * l + 1 for l in range(padded_l_max + 1)]
    features_even = features[0]

    features_even = features_even.reshape(
        features_even.shape[0], (padded_l_max + 1) * (padded_l_max + 1), features_even.shape[-1]
    )
    features_even = features_even.swapaxes(0, 1)
    stacked_features_from_even_uncoupled = (Us[0].T @ features_even.reshape(
        (padded_l_max + 1) * (padded_l_max + 1), features_even.shape[1] * features_even.shape[-1]
    )).reshape(
        (padded_l_max + 1) * (padded_l_max + 1), features_even.shape[1], features_even.shape[-1]
    )
    stacked_features_from_even_uncoupled = stacked_features_from_even_uncoupled.swapaxes(0, 1)
    features_from_even_uncoupled = torch.split(stacked_features_from_even_uncoupled, split_sizes, dim=1)

    features_odd = features[1]
    features_odd = features_odd.reshape(
        features_odd.shape[0], (padded_l_max + 1) * (padded_l_max + 1), features_odd.shape[-1]
    )
    features_odd = features_odd.swapaxes(0, 1)
    stacked_features_from_odd_uncoupled = (Us[1].T @ features_odd.reshape(
        (padded_l_max + 1) * (padded_l_max + 1), features_odd.shape[1] * features_odd.shape[-1]
    )).reshape(
        (padded_l_max + 1) * (padded_l_max + 1), features_odd.shape[1], features_odd.shape[-1]
    )
    stacked_features_from_odd_uncoupled = stacked_features_from_odd_uncoupled.swapaxes(0, 1)
    features_from_odd_uncoupled = torch.split(stacked_features_from_odd_uncoupled, split_sizes, dim=1)

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
