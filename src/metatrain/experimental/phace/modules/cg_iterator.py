from typing import Dict, List, Tuple

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from .cg import cg_combine_l1l2L
from .layers import EquivariantLinear, Linear
from .tensor_sum import EquivariantTensorAdd


class CGIterator(torch.nn.Module):
    # A high-level CG iterator, doing multiple iterations
    def __init__(
        self,
        k_max_l: List[int],
        number_of_iterations,
        cgs,
        irreps_in,
        requested_LS_string=None,
    ):
        super().__init__()
        self.k_max_l = k_max_l
        self.l_max = len(k_max_l) - 1
        self.number_of_iterations = number_of_iterations
        self.cgs = cgs
        self.requested_LS_string = requested_LS_string

        cg_iterations = []
        irreps_in_1 = irreps_in
        irreps_in_2 = irreps_in
        for n_iteration in range(self.number_of_iterations):
            if n_iteration == self.number_of_iterations - 1:
                requested_LS_string_now = requested_LS_string
            else:
                requested_LS_string_now = None
            cg_iterations.append(
                CGIterationAdd(
                    self.k_max_l,
                    cgs,
                    irreps_in_1,
                    irreps_in_2,
                    (n_iteration + 1, 1, n_iteration + 2),
                    requested_LS_string_now,
                )
            )
            irreps_out = cg_iterations[-1].irreps_out
            irreps_in_1 = irreps_out
        self.cg_iterations = torch.nn.ModuleList(cg_iterations)

        # equivariant linear mixers
        mixers = []
        for _ in range(self.number_of_iterations + 1):
            mixers.append(EquivariantLinear(irreps_in, k_max_l))
        self.mixers = torch.nn.ModuleList(mixers)

        self.irreps_out = self.cg_iterations[-1].irreps_out

    def forward(self, features: TensorMap) -> TensorMap:

        density = features
        mixed_densities = [mixer(density) for mixer in self.mixers]

        starting_density = mixed_densities[0]
        density_index = 1
        current_density = starting_density
        for iterator in self.cg_iterations:
            current_density = iterator(current_density, mixed_densities[density_index])
            density_index += 1

        return current_density


class CGIterationAdd(torch.nn.Module):
    # A single Clebsch-Gordan iteration, including a skip connection
    def __init__(
        self,
        k_max_l: List[int],
        cgs,
        irreps_in_1,
        irreps_in_2,
        nu_triplet,
        requested_LS_string=None,
    ):
        super().__init__()
        self.l_max = len(k_max_l) - 1
        self.cg_iteration = CGIteration(
            k_max_l, irreps_in_1, irreps_in_2, nu_triplet, cgs, requested_LS_string
        )
        self.irreps_out = self.cg_iteration.irreps_out

        common_irreps = [irrep for irrep in irreps_in_1 if irrep in self.irreps_out]
        self.adder = EquivariantTensorAdd(common_irreps, k_max_l)

    def forward(self, features_1: TensorMap, features_2: TensorMap):
        features_out = self.cg_iteration(features_1, features_2)
        features_out = self.adder(features_1, features_out)
        return features_out


class CGIteration(torch.nn.Module):
    # A single Clebsch-Gordan iteration (with contraction layers)
    def __init__(
        self,
        k_max_l: List[int],
        irreps_in_1: List[Tuple[int, int]],
        irreps_in_2: List[Tuple[int, int]],
        nu_triplet: Tuple[int, int, int],
        cgs,
        requested_LS_string=None,
    ):
        super().__init__()
        self.k_max_l = k_max_l
        self.l_max = len(k_max_l) - 1
        self.cgs = cgs
        self.irreps_out: List[Tuple[int, int]] = []
        self.requested_LS_string = requested_LS_string

        self.sizes_by_lam_sig: Dict[str, int] = {}
        for l1, s1 in irreps_in_1:
            for l2, s2 in irreps_in_2:
                for L in range(abs(l1 - l2), min(l1 + l2, self.l_max) + 1):
                    S = s1 * s2 * (-1) ** (l1 + l2 + L)
                    if S == -1:
                        continue
                    if self.requested_LS_string is not None:
                        if str(L) + "_" + str(S) != self.requested_LS_string:
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
                    Linear(size_LS, k_max_l[int(LS_string.split("_")[0])]),
                )
                for LS_string, size_LS in self.sizes_by_lam_sig.items()
            }
        )
        self.nu_out = nu_triplet[2]

    def forward(self, features_1: TensorMap, features_2: TensorMap):
        # handle dtype and device of the cgs
        if self.cgs["0_0_0"].device != features_1.device:
            self.cgs = {
                key: value.to(device=features_1.device)
                for key, value in self.cgs.items()
            }
        if self.cgs["0_0_0"].dtype != features_1.dtype:
            self.cgs = {
                key: value.to(dtype=features_1.block(0).values.dtype)
                for key, value in self.cgs.items()
            }

        # COULD DECREASE COST IF SYMMETRIC
        # Assume first and last dimension is the same for both
        results_by_lam_sig: Dict[str, List[torch.Tensor]] = {}
        for key_ls_1, block_ls_1 in features_1.items():
            l1s1 = key_ls_1.values
            l1, s1 = int(l1s1[1]), int(l1s1[2])
            for key_ls_2, block_ls_2 in features_2.items():
                l2s2 = key_ls_2.values
                l2, s2 = int(l2s2[1]), int(l2s2[2])
                min_size = min(block_ls_1.values.shape[2], block_ls_2.values.shape[2])
                tensor1 = block_ls_1.values[:, :, :min_size]
                tensor2 = block_ls_2.values[:, :, :min_size]
                tensor12 = tensor1.swapaxes(1, 2).unsqueeze(3) * tensor2.swapaxes(
                    1, 2
                ).unsqueeze(2)
                tensor12 = tensor12.reshape(tensor12.shape[0], tensor12.shape[1], -1)
                for L in range(abs(l1 - l2), min(l1 + l2, self.l_max) + 1):
                    S = int(s1 * s2 * (-1) ** (l1 + l2 + L))
                    if self.requested_LS_string is not None:
                        if str(L) + "_" + str(S) != self.requested_LS_string:
                            continue
                    result = cg_combine_l1l2L(
                        tensor12, self.cgs[str(l1) + "_" + str(l2) + "_" + str(L)]
                    )
                    if (str(L) + "_" + str(S)) not in results_by_lam_sig:
                        results_by_lam_sig[(str(L) + "_" + str(S))] = [result]
                    else:
                        results_by_lam_sig[(str(L) + "_" + str(S))].append(result)

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
                    samples=features_1.block({"o3_lambda": 0, "o3_sigma": 1}).samples,
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

        return TensorMap(
            keys=Labels(
                names=["nu", "o3_lambda", "o3_sigma"],
                values=torch.tensor(keys, device=blocks[0].values.device),
            ),
            blocks=blocks,
        )
