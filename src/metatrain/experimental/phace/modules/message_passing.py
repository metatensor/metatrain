from typing import Dict, List, Tuple

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from .cg import cg_combine_l1l2L
from .layers import Linear
from .radial_basis import RadialBasis

from .tensor_sum import EquivariantTensorAdd
import metatensor.torch


class DummyAdder(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, tmap_1: metatensor.torch.TensorMap, tmap_2: metatensor.torch.TensorMap) -> metatensor.torch.TensorMap:
        return metatensor.torch.TensorMap(
            keys=Labels(
                names=["dummy"],
                values=torch.empty(1, 1)
            ),
            blocks=[]
        )


class InvariantMessagePasser(torch.nn.Module):

    def __init__(self, hypers: Dict, all_species: List[int], mp_scaling, disable_nu_0) -> None:
        super().__init__()

        self.all_species = all_species
        hypers["radial_basis"]["r_cut"] = hypers["cutoff"]
        hypers["radial_basis"]["n_element_channels"] = hypers["n_element_channels"]
        self.radial_basis_calculator = RadialBasis(hypers["radial_basis"], all_species)
        self.n_max_l = self.radial_basis_calculator.n_max_l
        self.k_max_l = [hypers["n_element_channels"] * n_max for n_max in self.n_max_l]
        self.l_max = len(self.n_max_l) - 1
        self.irreps_out = [(l, 1) for l in range(self.l_max + 1)]

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
        for l in range(self.l_max + 1):
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
        if not self.disable_nu_0:
            pooled_result = self.adder(pooled_result, initial_center_embedding)
        return pooled_result


class EquivariantMessagePasser(torch.nn.Module):

    def __init__(
        self,
        hypers: Dict,
        all_species: List[int],
        irreps_in_features: List[Tuple[int, int]],
        nu_triplet: Tuple[int, int, int],
        cgs,
        mp_scaling,
    ) -> None:
        super().__init__()

        # TODO: extract the radial basis calculator from this class
        # sparing us the need for all_species

        self.all_species = all_species
        hypers["radial_basis"]["r_cut"] = hypers["cutoff"]
        hypers["radial_basis"]["n_element_channels"] = hypers["n_element_channels"]
        self.radial_basis_calculator = RadialBasis(hypers["radial_basis"], all_species)
        self.n_max_l = self.radial_basis_calculator.n_max_l
        self.k_max_l = [hypers["n_element_channels"] * n_max for n_max in self.n_max_l]
        self.l_max = len(self.n_max_l) - 1

        self.cgs = cgs
        self.irreps_out = []
        self.irreps_in_vector_expansion = [(l, 1) for l in range(self.l_max + 1)]
        self.irreps_in_features = irreps_in_features

        self.sizes_by_lam_sig: Dict[str, int] = {}
        for l1, s1 in self.irreps_in_vector_expansion:
            for l2, s2 in self.irreps_in_features:
                for L in range(abs(l1 - l2), min(l1 + l2, self.l_max) + 1):
                    S = s1 * s2 * (-1) ** (l1 + l2 + L)
                    if S == -1:
                        continue
                    if (L, S) not in self.irreps_out:
                        self.irreps_out.append((L, S))
                    larger_l = max(l1, l2)
                    size = self.k_max_l[larger_l]
                    if (str(L) + "_" + str(S)) not in self.sizes_by_lam_sig:
                        self.sizes_by_lam_sig[(str(L) + "_" + str(S))] = size
                    else:
                        self.sizes_by_lam_sig[(str(L) + "_" + str(S))] += size

        # Register linear layers for contraction:
        self.linear_contractions = torch.nn.ModuleDict(
            {
                LS_string: torch.nn.Sequential(
                    Linear(size_LS, self.k_max_l[int(LS_string.split("_")[0])]),
                )
                for LS_string, size_LS in self.sizes_by_lam_sig.items()
            }
        )
        self.nu_out = nu_triplet[2]

        common_irreps = [irrep for irrep in self.irreps_out if irrep in irreps_in_features]
        self.adder = EquivariantTensorAdd(common_irreps, self.k_max_l)
        # self.adder = EquivariantTensorAdd()

        self.mp_scaling = mp_scaling

        self.irreps_out = list(set(self.irreps_out + self.irreps_in_features))
        

    def forward(
        self,
        r: TensorBlock,
        sh: TensorMap,
        centers,
        neighbors,
        n_atoms: int,
        features: TensorMap,
        samples: Labels,  # TODO: can this go?
    ) -> TensorMap:
        # handle dtype and device of the cgs
        if self.cgs["0_0_0"].device != r.device:
            self.cgs = {
                key: value.to(device=r.device) for key, value in self.cgs.items()
            }
        if self.cgs["0_0_0"].dtype != r.dtype:
            self.cgs = {
                key: value.to(dtype=r.values.dtype) for key, value in self.cgs.items()
            }

        radial_basis = self.radial_basis_calculator(r.values.squeeze(-1), r.samples)

        vector_expansion = [
            sh.block({"o3_lambda": l}).values * radial_basis[l].unsqueeze(1)
            for l in range(self.l_max + 1)
        ]

        results_by_lam_sig: Dict[str, List[torch.Tensor]] = {}
        for l1, s1 in self.irreps_in_vector_expansion:
            block_ls_1 = vector_expansion[l1]
            for key_ls_2, block_ls_2 in features.items():
                l2s2 = key_ls_2.values
                l2, s2 = int(l2s2[1]), int(l2s2[2])
                min_size = min(block_ls_1.shape[2], block_ls_2.values.shape[2])
                tensor1 = block_ls_1[:, :, :min_size]
                tensor2 = block_ls_2.values[neighbors, :, :min_size]
                tensor12 = tensor1.swapaxes(1, 2).unsqueeze(3) * tensor2.swapaxes(
                    1, 2
                ).unsqueeze(2)
                tensor12 = tensor12.reshape(tensor12.shape[0], tensor12.shape[1], tensor12.shape[2]*tensor12.shape[3])
                for L in range(abs(l1 - l2), min(l1 + l2, self.l_max) + 1):
                    S = int(s1 * s2 * (-1) ** (l1 + l2 + L))
                    result = cg_combine_l1l2L(
                        tensor12, self.cgs[str(l1) + "_" + str(l2) + "_" + str(L)]
                    )
                    pooled_result = torch.zeros(
                        (n_atoms, result.shape[1], result.shape[2]),
                        device=result.device,
                        dtype=result.dtype,
                    )
                    pooled_result.index_add_(
                        dim=0,
                        index=centers,
                        source=result,
                    )
                    pooled_result = pooled_result
                    if (str(L) + "_" + str(S)) not in results_by_lam_sig:
                        results_by_lam_sig[(str(L) + "_" + str(S))] = [pooled_result]
                    else:
                        results_by_lam_sig[(str(L) + "_" + str(S))].append(
                            pooled_result
                        )

        compressed_results_by_lam_sig: Dict[str, torch.Tensor] = {}
        for LS_string, linear_LS in self.linear_contractions.items():
            split_LS_string = LS_string.split("_")
            L = int(split_LS_string[0])
            S = int(split_LS_string[1])
            concatenated_tensor = torch.concatenate(
                results_by_lam_sig[LS_string], dim=2
            )
            compressed_tensor = linear_LS(concatenated_tensor)
            compressed_results_by_lam_sig[(str(L) + "_" + str(S))] = compressed_tensor

        keys: List[List[int]] = []
        blocks: List[TensorBlock] = []
        for LS_string, compressed_tensor_LS in compressed_results_by_lam_sig.items():
            split_LS_string = LS_string.split("_")
            L = int(split_LS_string[0])
            S = int(split_LS_string[1])
            blocks.append(
                TensorBlock(
                    values=compressed_tensor_LS,
                    samples=features.block({"o3_lambda": 0, "o3_sigma": 1}).samples,
                    components=[
                        Labels(
                            names=["o3_mu"],
                            values=torch.arange(
                                start=-L,
                                end=L + 1,
                                dtype=torch.int,
                                device=compressed_tensor_LS.device,
                            ).reshape(2 * L + 1, 1),
                        ).to(compressed_tensor_LS.device)
                    ],
                    properties=Labels(
                        "properties",
                        torch.arange(
                            compressed_tensor_LS.shape[2],
                            dtype=torch.int,
                            device=compressed_tensor_LS.device,
                        ).unsqueeze(-1),
                    ),
                )
            )
            keys.append([self.nu_out, L, S])

        pooled_result = TensorMap(
            keys=Labels(
                names=["nu", "o3_lambda", "o3_sigma"],
                values=torch.tensor(keys, device=blocks[0].values.device),
            ),
            blocks=blocks,
        )
        pooled_result = metatensor.torch.multiply(pooled_result, self.mp_scaling)
        pooled_result = self.adder(pooled_result, features)
        return pooled_result
