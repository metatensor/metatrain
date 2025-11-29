from typing import List

import torch

from .layers import LinearList as Linear
from .radial_mlp import MLPRadialBasis
from .tensor_product import (
    combine_uncoupled_features,
    couple_features,
    uncouple_features,
)


class InvariantMessagePasser(torch.nn.Module):
    # performs invariant message passing with linear contractions
    def __init__(
        self, all_species: List[int], mp_scaling, disable_nu_0, n_max_l, num_element_channels
    ) -> None:
        super().__init__()

        self.all_species = all_species
        self.radial_basis_mlp = MLPRadialBasis(n_max_l, num_element_channels)
        self.n_max_l = n_max_l
        # self.k_max_l = [128, 128, 128]
        self.k_max_l = [
            num_element_channels * n_max for n_max in self.n_max_l
        ]
        self.l_max = len(self.n_max_l) - 1
        self.irreps_out = [(l, 1) for l in range(self.l_max + 1)]  # noqa: E741

        self.mp_scaling = mp_scaling
        self.disable_nu_0 = disable_nu_0

    def forward(
        self,
        radial_basis: List[torch.Tensor],
        spherical_harmonics: List[torch.Tensor],
        centers,
        neighbors,
        n_atoms: int,
        initial_center_embedding,
    ) -> List[torch.Tensor]:
        # TODO: extract radial basis calculation to a separate module
        # (e.g. vector expansion) and use the splines once
        radial_basis = self.radial_basis_mlp(radial_basis)

        density = []
        for l in range(self.l_max + 1):  # noqa: E741
            spherical_harmonics_l = spherical_harmonics[l]
            radial_basis_l = radial_basis[l]
            density_l = torch.zeros(
                (n_atoms, spherical_harmonics_l.shape[1], radial_basis_l.shape[1]),
                device=radial_basis_l.device,
                dtype=radial_basis_l.dtype,
            )
            density_l.index_add_(
                dim=0,
                index=centers,
                source=spherical_harmonics_l.unsqueeze(2)
                * radial_basis_l.unsqueeze(1)
                * initial_center_embedding[neighbors][
                    :, :, : radial_basis_l.shape[1]
                ],
            )
            density.append(density_l * self.mp_scaling)

        # TODO: add linear layers here?

        if not self.disable_nu_0:
            density[0] = density[0] + initial_center_embedding
        
        return density


class EquivariantMessagePasser(torch.nn.Module):
    # performs equivariant message passing with linear contractions
    def __init__(
        self,
        n_max_l,
        num_element_channels,
        tensor_product,
        mp_scaling,
    ) -> None:
        super().__init__()

        self.n_max_l = list(n_max_l)
        # print(self.n_max_l)
        # print(num_element_channels)
        self.k_max_l = [
            num_element_channels * n_max for n_max in self.n_max_l
        ]
        # self.k_max_l = [128, 128, 128]
        self.l_max = len(self.n_max_l) - 1

        self.k_max_l_max = [0] * (self.l_max + 1)
        previous = 0
        for l in range(self.l_max, -1, -1):
            self.k_max_l_max[l] = self.k_max_l[l] - previous
            previous = self.k_max_l[l]

        self.mp_scaling = mp_scaling
        self.padded_l_list = tensor_product.padded_l_list
        self.U_dict = tensor_product.U_dict

        self.linear = Linear(self.k_max_l)

        self.radial_basis_mlp = MLPRadialBasis(n_max_l, num_element_channels)

    def forward(
        self,
        radial_basis: List[torch.Tensor],
        spherical_harmonics: List[torch.Tensor],
        centers,
        neighbors,
        features: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        device = features[0].device
        if self.U_dict[0].device != device:
            self.U_dict = {key: U.to(device) for key, U in self.U_dict.items()}
        dtype = features[0].dtype
        if self.U_dict[0].dtype != dtype:
            self.U_dict = {key: U.to(dtype) for key, U in self.U_dict.items()}

        radial_basis = self.radial_basis_mlp(radial_basis)
        vector_expansion = [
            spherical_harmonics[l].unsqueeze(2) * radial_basis[l].unsqueeze(1)
            for l in range(self.l_max + 1)  # noqa: E741
        ]

        split_vector_expansion: List[List[torch.Tensor]] = []
        for l in range(self.l_max, -1, -1):
            lower_bound = self.k_max_l[l + 1] if l < self.l_max else 0
            upper_bound = self.k_max_l[l]
            split_vector_expansion = [
                [
                    vector_expansion[lp][:, :, lower_bound:upper_bound]
                    for lp in range(l + 1)
                ]
            ] + split_vector_expansion

        uncoupled_vector_expansion = []
        for l in range(self.l_max + 1):
            uncoupled_vector_expansion.append(
                uncouple_features(
                    split_vector_expansion[l],
                    self.U_dict[self.padded_l_list[l]],
                    self.padded_l_list[l],
                )
            )

        split_features: List[List[torch.Tensor]] = []
        for l in range(self.l_max, -1, -1):
            lower_bound = self.k_max_l[l + 1] if l < self.l_max else 0
            upper_bound = self.k_max_l[l]
            split_features = [
                [features[lp][:, :, lower_bound:upper_bound] for lp in range(l + 1)]
            ] + split_features

        uncoupled_features = []
        for l in range(self.l_max + 1):
            uncoupled_features.append(
                uncouple_features(
                    split_features[l],
                    self.U_dict[self.padded_l_list[l]],
                    self.padded_l_list[l],
                )
            )

        n_atoms = features[0].shape[0]

        indexed_features = []
        for feature in uncoupled_features:
            indexed_features.append(feature[neighbors])

        # TODO: maybe it would be a good idea to break these up to limit memory usage
        combined_features = combine_uncoupled_features(
            uncoupled_vector_expansion, indexed_features
        )

        combined_features_pooled = []
        for f in combined_features:
            combined_features_pooled.append(
                torch.zeros(
                    (n_atoms,) + f.shape[1:],
                    device=f.device,
                    dtype=f.dtype,
                ),
            )
            combined_features_pooled[-1].index_add_(
                dim=0,
                index=centers,
                source=f,
            )

        coupled_features: List[List[torch.Tensor]] = []
        for l in range(self.l_max + 1):
            coupled_features.append(
                couple_features(
                    combined_features_pooled[l],
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

        # apply mp_scaling
        combined_features_pooled = [
            (f * self.mp_scaling) for f in concatenated_coupled_features
        ]

        features_out = self.linear(concatenated_coupled_features)
        features_out = [f + fo for f, fo in zip(features, features_out, strict=False)]
        return features_out
