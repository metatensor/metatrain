from typing import Any, Callable, Optional

import torch
from e3nn import o3
from e3nn.nn import Activation
from mace.modules.blocks import LinearReadoutBlock
from mace.modules.wrapper_ops import (
    CuEquivarianceConfig,
    Linear,
)


class NonLinearHead(torch.nn.Module):
    """Generic non-linear head with two linear layers and an activation in between.

    The activation is only applied to the irreps with l>0. Therefore, if there is no
    irreps with l=0, this module is a bit stupid (it just applies two linear layers
    one after the other). But for consistency, it is better to always use this
    head.

    The module uses MACE's Linear wrapper to support cuequivariance or whatever
    other optimizations that MACE provides.

    :param irreps_in: Input irreps.
    :param MLP_irreps: Irreps for the hidden layer of the MLP (only for l=0).
    :param irreps_out: Output irreps.
    :param gate: Activation function to use for the l=0 irreps.
    :param cueq_config: Configuration for CUDA equivariant operations.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        irreps_out: o3.Irreps,
        gate: Optional[Callable],
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        super().__init__()
        # Get the l values present in the output irreps, so that we can filter
        # out the irreps that are not really used in the last layer, therefore
        # having only the last layer features that are truly used.
        output_ls = set(ir.ir.l for ir in irreps_out)

        self.hidden_irreps = sum(
            [
                ir if ir.ir.l > 0 else MLP_irreps
                for ir in irreps_in
                if ir.ir.l in output_ls
            ],
            o3.Irreps(""),
        )
        gates = [None if ir.ir.l > 0 else gate for ir in self.hidden_irreps]
        self.linear_1 = Linear(
            irreps_in=irreps_in, irreps_out=self.hidden_irreps, cueq_config=cueq_config
        )
        self.non_linearity = Activation(irreps_in=self.hidden_irreps, acts=gates)
        self.linear_2 = Linear(
            irreps_in=self.hidden_irreps, irreps_out=irreps_out, cueq_config=cueq_config
        )

        self.last_layer_features_irreps = self.hidden_irreps
        self.last_layer_features = torch.empty(0)  # To be replaced at forward pass

    def forward(
        self,
        node_features: torch.Tensor,
        node_energies: Optional[torch.Tensor] = None,
        compute_llf: bool = False,
    ) -> torch.Tensor:
        node_features = self.non_linearity(self.linear_1(node_features))
        self.last_layer_features = node_features
        return self.linear_2(node_features)


# ---------------------------------------------------------
# Internal MACE Head Wrapper to extract last layer features
# ---------------------------------------------------------


def readout_is_linear(obj: Any):
    if isinstance(obj, torch.jit.RecursiveScriptModule):
        return obj.original_name == "LinearReadoutBlock"
    else:
        return isinstance(obj, LinearReadoutBlock)


class MACEHeadWrapper(torch.nn.Module):
    """Wrapper around MACE readout heads to extract last layer features (LLF)."""

    def __init__(
        self,
        readouts: torch.nn.ModuleList,
        per_layer_irreps: o3.Irreps,
    ):
        super().__init__()

        self.per_layer_irreps = per_layer_irreps
        self.per_layer_dims = [ir.dim for ir in self.per_layer_irreps]
        self.per_layer_scalars = [ir.count((0, 1)) for ir in self.per_layer_irreps]
        features_irreps = sum(self.per_layer_irreps, o3.Irreps())

        self.last_layer_features_irreps = features_irreps.count((0, 1)) * o3.Irrep(0, 1)
        self.last_layer_features = torch.empty(0)  # To be replaced at forward pass

        self.mace_llf_extractors = torch.nn.ModuleList(
            [
                torch.nn.Identity()
                if readout_is_linear(readout)
                else torch.nn.Sequential(readout.linear_1, readout.non_linearity)
                for readout in readouts
            ]
        )

    def forward(
        self,
        node_features: torch.Tensor,
        node_energies: torch.Tensor,
        compute_llf: bool = False,
    ) -> torch.Tensor:
        node_energies = node_energies.to(dtype=node_features.dtype).reshape(-1, 1)
        if compute_llf:
            per_layer_features = torch.split(node_features, self.per_layer_dims, dim=-1)

            ll_feats_list = [
                extractor(per_layer_features[i])[:, : self.per_layer_scalars[i]]
                for i, extractor in enumerate(self.mace_llf_extractors)
            ]

            # Aggregate node features
            ll_features = torch.cat(ll_feats_list, dim=-1)

            self.last_layer_features = ll_features

        return node_energies
