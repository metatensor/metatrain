import copy
from typing import Dict, List, Tuple

import metatensor.torch
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from .cg import cg_combine_l1l2L
from .layers import LinearList as Linear
from .radial_basis import RadialBasis
from .tensor_product import combine_uncoupled_features, uncouple_features
from .tensor_sum import EquivariantTensorAdd


class DummyAdder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tmap_1: TensorMap, tmap_2: TensorMap) -> TensorMap:
        return TensorMap(
            keys=Labels(names=["dummy"], values=torch.empty(1, 1)), blocks=[]
        )


class InvariantMessagePasser(torch.nn.Module):
    # performs invariant message passing with linear contractions
    def __init__(
        self, hypers: Dict, all_species: List[int], mp_scaling, disable_nu_0
    ) -> None:
        super().__init__()

        self.all_species = all_species
        radial_basis_hypers = copy.deepcopy(hypers["radial_basis"])
        radial_basis_hypers["cutoff"] = hypers["cutoff"]
        radial_basis_hypers["num_element_channels"] = hypers["num_element_channels"]
        radial_basis_hypers["cutoff_width"] = hypers["cutoff_width"]
        self.radial_basis_calculator = RadialBasis(radial_basis_hypers, all_species)
        self.n_max_l = self.radial_basis_calculator.n_max_l
        self.k_max_l = [
            hypers["num_element_channels"] * n_max for n_max in self.n_max_l
        ]
        self.l_max = len(self.n_max_l) - 1
        self.irreps_out = [(l, 1) for l in range(self.l_max + 1)]  # noqa: E741

        self.mp_scaling = mp_scaling

        if not disable_nu_0:
            self.adder = EquivariantTensorAdd([(0, 1)], self.k_max_l)
        else:
            self.adder = DummyAdder()
        self.disable_nu_0 = disable_nu_0

    def forward(
        self,
        r: TensorBlock,
        sh: TensorMap,
        centers,
        neighbors,
        n_atoms: int,
        initial_center_embedding: TensorMap,
        samples: Labels,  # TODO: can this go?
    ) -> TensorMap:

        # TODO: extract radial basis calculation to a separate module
        # (e.g. vector expansion) and use the splines once
        radial_basis = self.radial_basis_calculator(r.values.squeeze(-1), r.samples)

        labels: List[List[int]] = []
        blocks: List[TensorBlock] = []
        for l in range(self.l_max + 1):  # noqa: E741
            spherical_harmonics_l = sh.block({"o3_lambda": l}).values
            radial_basis_l = radial_basis[l]
            densities_l = torch.zeros(
                (n_atoms, spherical_harmonics_l.shape[1], radial_basis_l.shape[1]),
                device=radial_basis_l.device,
                dtype=radial_basis_l.dtype,
            )
            densities_l.index_add_(
                dim=0,
                index=centers,
                source=spherical_harmonics_l
                * radial_basis_l.unsqueeze(1)
                * initial_center_embedding.block().values[neighbors][
                    :, :, : radial_basis_l.shape[1]
                ],
            )
            labels.append([1, l, 1])
            blocks.append(
                TensorBlock(
                    values=densities_l,
                    samples=samples,
                    components=sh.block({"o3_lambda": l}).components,
                    properties=Labels(
                        "properties",
                        torch.arange(
                            densities_l.shape[2],
                            dtype=torch.int,
                            device=densities_l.device,
                        ).unsqueeze(-1),
                    ),
                )
            )

        pooled_result = TensorMap(
            keys=Labels(
                names=["nu", "o3_lambda", "o3_sigma"],
                values=torch.tensor(labels, dtype=torch.int32),
            ).to(device=initial_center_embedding.device),
            blocks=blocks,
        )
        pooled_result = metatensor.torch.multiply(pooled_result, self.mp_scaling)

        # TODO: add linear layers here?

        if not self.disable_nu_0:
            pooled_result = self.adder(pooled_result, initial_center_embedding)
        return pooled_result


class EquivariantMessagePasser(torch.nn.Module):
    # performs equivariant message passing with linear contractions
    def __init__(
        self,
        hypers: Dict,
        all_species: List[int],
        padded_l_list,
        mp_scaling,
    ) -> None:
        super().__init__()

        # TODO: extract the radial basis calculator from this class
        # sparing us the need for all_species

        self.all_species = all_species
        radial_basis_hypers = copy.deepcopy(hypers["radial_basis"])
        radial_basis_hypers["cutoff"] = hypers["cutoff"]
        radial_basis_hypers["num_element_channels"] = hypers["num_element_channels"]
        radial_basis_hypers["cutoff_width"] = hypers["cutoff_width"]
        self.radial_basis_calculator = RadialBasis(radial_basis_hypers, all_species)
        self.n_max_l = self.radial_basis_calculator.n_max_l
        self.k_max_l = [
            hypers["num_element_channels"] * n_max for n_max in self.n_max_l
        ]
        self.l_max = len(self.n_max_l) - 1

        self.k_max_l_max = [0] * (self.l_max + 1)
        previous = 0
        for l in range(self.l_max, -1, -1):
            self.k_max_l_max[l] = self.k_max_l[l] - previous
            previous = self.k_max_l[l]

        self.mp_scaling = mp_scaling
        self.padded_l_list = padded_l_list

        self.linear = Linear(self.k_max_l_max)

    def forward(
        self,
        r: TensorBlock,
        sh: TensorMap,
        centers,
        neighbors,
        features: List[Tuple[torch.Tensor, torch.Tensor]],
        U_dict_parity: Dict[str, torch.Tensor],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:

        radial_basis = self.radial_basis_calculator(r.values.squeeze(-1), r.samples)
        vector_expansion = [
            sh.block({"o3_lambda": l}).values * radial_basis[l].unsqueeze(1)
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

        uncoupled_vector_expansion: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for l in range(self.l_max + 1):
            uncoupled_vector_expansion.append(
                uncouple_features(
                    split_vector_expansion[l],
                    (U_dict_parity[f"{self.padded_l_list[l]}_{1}"], U_dict_parity[f"{self.padded_l_list[l]}_{-1}"]),
                    self.padded_l_list[l],
                )
            )

        n_atoms = features[0][0].shape[0]

        indexed_features: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for feature_even, feature_odd in features:
            indexed_features.append((feature_even[neighbors], feature_odd[neighbors]))

        # TODO: maybe it would be a good idea to break these up to limit memory usage
        combined_features = combine_uncoupled_features(
            uncoupled_vector_expansion, indexed_features
        )

        combined_features_pooled: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for cfe, cfo in combined_features:
            combined_features_pooled.append(
                (
                    torch.zeros(
                        (n_atoms,) + cfe.shape[1:],
                        device=cfe.device,
                        dtype=cfe.dtype,
                    ),
                    torch.zeros(
                        (n_atoms,) + cfo.shape[1:],
                        device=cfo.device,
                        dtype=cfo.dtype,
                    )
                )
            )
            combined_features_pooled[-1][0].index_add_(
                dim=0,
                index=centers,
                source=cfe,
            )
            combined_features_pooled[-1][1].index_add_(
                dim=0,
                index=centers,
                source=cfo,
            )

        # apply mp_scaling
        combined_features_pooled = [
            (fe * self.mp_scaling, fo * self.mp_scaling) for fe, fo in combined_features_pooled
        ]

        features_out = self.linear(combined_features_pooled)
        features_out = [
            (f1e + foe, f1o + foo) for (f1e, f1o), (foe, foo) in zip(features, features_out)
        ]

        return features_out
