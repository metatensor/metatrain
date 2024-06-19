# this implements a CG iterator that goes through all possible paths
# for each nu

from typing import Dict, List, Tuple

import torch
from metatensor.torch import TensorMap

from .cg_iterator import CGIteration
from .tensor_sum import EquivariantTensorAdd


class GeneralizedCGIterator(torch.nn.Module):
    # dumb indexing and data structures due to torchscript
    def __init__(
        self,
        k_max_l: List[int],
        nu_max: int,
        cgs,
        irreps_in: Dict[int, List[Tuple[int, int]]],
        # requested_LS_string=None
    ):
        super().__init__()
        self.k_max_l = k_max_l
        self.l_max = len(k_max_l) - 1
        self.nu_max = nu_max
        self.cgs = cgs
        # self.requested_LS_string = requested_LS_string

        irreps_out = {1: irreps_in[1]}
        # Do all possible paths
        cg_iterations: List[Dict[Tuple[int, int], torch.nn.Module]] = [
            {} for _ in range(2, self.nu_max + 1)
        ]
        for nu in range(2, self.nu_max + 1):
            irreps_out[nu] = []
            for i in range(1, nu // 2 + 1):
                cg_iterations[nu - 2][f"{i}_{nu-i}"] = CGIteration(
                    k_max_l=k_max_l,
                    irreps_in_1=irreps_out[i],
                    irreps_in_2=irreps_out[nu - i],
                    nu_triplet=(i, nu - i, nu),
                    cgs=cgs,
                )
                irreps_out[nu].append(cg_iterations[nu - 2][f"{i}_{nu-i}"].irreps_out)
            # merge all irreps_out for this nu
            irreps_out[nu] = merge_irreps(*irreps_out[nu])

        self.irreps_out = irreps_out
        self.cg_iterations = torch.nn.ModuleList(
            [torch.nn.ModuleDict(cg_iteration) for cg_iteration in cg_iterations]
        )
        self.adder = EquivariantTensorAdd()

    def forward(self, in_features: List[TensorMap]) -> List[TensorMap]:
        # handle dtype and device of the cgs
        if self.cgs["0_0_0"].device != in_features[0].device:
            self.cgs = {
                key: value.to(device=in_features[0].device)
                for key, value in self.cgs.items()
            }
        if self.cgs["0_0_0"].dtype != in_features[0].dtype:
            self.cgs = {
                key: value.to(dtype=in_features[0].block(0).values.dtype)
                for key, value in self.cgs.items()
            }

        out_features = [in_features[0], self.adder(in_features[0], in_features[1])]

        for idx, moduledict in enumerate(self.cg_iterations):
            nu = idx + 2
            out_features_list: List[TensorMap] = []
            for key, module in moduledict.items():
                i, nu_minus_i = int(key.split("_")[0]), int(key.split("_")[1])
                out_features_list.append(
                    module(out_features[i], out_features[nu_minus_i])
                )
            # sum all the outputs for this nu, but the adder can only take two inputs
            while len(out_features_list) > 1:
                popped = out_features_list.pop(-1)
                out_features_list[-1] = self.adder(out_features_list[-1], popped)
            out_features.append(out_features_list[0])
            out_features[nu] = self.adder(out_features[nu], out_features[nu - 1])

        return out_features


def merge_irreps(*args) -> List[Tuple[int, int]]:
    # merges irreps_1 and irreps_2, only keeping unique elements
    return list(set([irrep for irreps in args for irrep in irreps]))
