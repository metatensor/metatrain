from typing import Dict, List

import torch

from .layers import EquivariantRMSNorm
from .layers import LinearList as Linear
from .radial_mlp import MLPRadialBasis
from .tensor_product import (
    tensor_product,
    uncouple_features_all,
)


class InvariantMessagePasser(torch.nn.Module):
    # performs invariant message passing with linear contractions
    def __init__(
        self,
        all_species: List[int],
        message_scaling,
        n_max_l,
        k_max_l,
        radial_mlp_depth,
        mlp_width_factor,
    ) -> None:
        super().__init__()

        self.all_species = all_species
        self.radial_basis_mlp = MLPRadialBasis(
            n_max_l, k_max_l, depth=radial_mlp_depth, width_factor=mlp_width_factor
        )
        self.n_max_l = n_max_l
        self.k_max_l = k_max_l
        self.l_max = len(self.n_max_l) - 1
        self.irreps_out = [(l, 1) for l in range(self.l_max + 1)]  # noqa: E741

        # Register message_scaling as a buffer for efficiency
        self.register_buffer("message_scaling", torch.tensor(message_scaling))

    def forward(
        self,
        radial_basis: List[torch.Tensor],
        spherical_harmonics: List[torch.Tensor],
        centers,
        neighbors,
        n_atoms: int,
        initial_center_embedding,
    ) -> List[torch.Tensor]:
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
            density.append(density_l * self.message_scaling)

        density[0] = density[0] + initial_center_embedding
        return density


class EquivariantMessagePasser(torch.nn.Module):
    # performs equivariant message passing with a norm operation and linear contractions
    def __init__(
        self,
        n_max_l,
        k_max_l,
        message_scaling,
        radial_mlp_depth: int = 3,
        mlp_width_factor: int = 4,
    ) -> None:
        super().__init__()

        self.n_max_l = list(n_max_l)
        self.k_max_l = k_max_l
        self.l_max = len(self.n_max_l) - 1

        # Register message_scaling as a buffer for efficiency
        self.register_buffer("message_scaling", torch.tensor(message_scaling))
        self.padded_l_list = [2 * ((l + 1) // 2) for l in range(self.l_max + 1)]  # noqa: E741

        self.linear_in = Linear(self.k_max_l)
        self.rmsnorm = EquivariantRMSNorm(self.k_max_l)
        self.linear_out = Linear(self.k_max_l)

        self.radial_basis_mlp = MLPRadialBasis(
            n_max_l, k_max_l, depth=radial_mlp_depth, width_factor=mlp_width_factor
        )

    def forward(
        self,
        radial_basis: List[torch.Tensor],
        spherical_harmonics: List[torch.Tensor],
        centers,
        neighbors,
        features: List[torch.Tensor],
        U_dict: Dict[int, torch.Tensor],
    ) -> List[torch.Tensor]:
        n_atoms = features[0].shape[0]

        ### 1. Norm and linear #####
        features_in = features
        features = self.rmsnorm(features)
        features = self.linear_in(features, U_dict)

        ##### 2. Compute radial basis, vector expansion in the spherical basis, and
        # transform the vector expansion to the uncoupled basis #####
        radial_basis = self.radial_basis_mlp(radial_basis)
        vector_expansion = [
            spherical_harmonics[l].unsqueeze(2) * radial_basis[l].unsqueeze(1)
            for l in range(self.l_max + 1)  # noqa: E741
        ]
        uncoupled_vector_expansion = uncouple_features_all(
            vector_expansion, self.k_max_l, U_dict, self.l_max, self.padded_l_list
        )

        ##### 3. Message passing #####
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
        combined_features_pooled = [
            (f * self.message_scaling) for f in combined_features_pooled
        ]

        ##### 4. Linear and residual connection #####
        features_out = self.linear_out(combined_features_pooled, U_dict)
        features_out = [
            fi + fo for fi, fo in zip(features_in, features_out, strict=True)
        ]

        return features_out
