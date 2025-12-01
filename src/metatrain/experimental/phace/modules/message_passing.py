from typing import Dict, List

import torch

from .layers import LinearList as Linear
from .radial_mlp import MLPRadialBasis
from .tensor_product import (
    split_up_features,
    tensor_product,
    uncouple_features,
)


class InvariantMessagePasser(torch.nn.Module):
    # performs invariant message passing with linear contractions
    def __init__(
        self, all_species: List[int], mp_scaling, disable_nu_0, n_max_l, k_max_l
    ) -> None:
        super().__init__()

        self.all_species = all_species
        self.radial_basis_mlp = MLPRadialBasis(n_max_l, k_max_l)
        self.n_max_l = n_max_l
        self.k_max_l = k_max_l
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
                * initial_center_embedding[neighbors][:, :, : radial_basis_l.shape[1]],
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
        k_max_l,
        mp_scaling,
        spherical_linear_layers,
    ) -> None:
        super().__init__()

        self.n_max_l = list(n_max_l)
        self.k_max_l = k_max_l
        self.l_max = len(self.n_max_l) - 1

        self.mp_scaling = mp_scaling
        self.padded_l_list = [2 * ((l + 1) // 2) for l in range(self.l_max + 1)]  # noqa: E741

        self.linear = Linear(self.k_max_l, spherical_linear_layers)

        self.radial_basis_mlp = MLPRadialBasis(n_max_l, k_max_l)

    def forward(
        self,
        radial_basis: List[torch.Tensor],
        spherical_harmonics: List[torch.Tensor],
        centers,
        neighbors,
        features: List[torch.Tensor],
        U_dict: Dict[int, torch.Tensor],
    ) -> List[torch.Tensor]:
        radial_basis = self.radial_basis_mlp(radial_basis)
        vector_expansion = [
            spherical_harmonics[l].unsqueeze(2) * radial_basis[l].unsqueeze(1)
            for l in range(self.l_max + 1)  # noqa: E741
        ]

        split_vector_expansion = split_up_features(vector_expansion, self.k_max_l)
        uncoupled_vector_expansion = []
        for l in range(self.l_max + 1):  # noqa: E741
            uncoupled_vector_expansion.append(
                uncouple_features(
                    split_vector_expansion[l],
                    U_dict[self.padded_l_list[l]],
                    self.padded_l_list[l],
                )
            )

        n_atoms = features[0].shape[0]

        indexed_features = []
        for feature in features:
            indexed_features.append(feature[neighbors])

        combined_features = tensor_product(uncoupled_vector_expansion, indexed_features)

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

        # apply mp_scaling
        combined_features_pooled = [
            (f * self.mp_scaling) for f in combined_features_pooled
        ]

        features_out = self.linear(combined_features_pooled, U_dict)
        features_out = [f + fo for f, fo in zip(features, features_out, strict=False)]
        return features_out
