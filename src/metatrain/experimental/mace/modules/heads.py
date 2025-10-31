from typing import Callable, Optional

import torch
from e3nn import o3
from e3nn.nn import Activation
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
        self.hidden_irreps = sum(
            [str(ir) if ir.ir.l > 0 else MLP_irreps for ir in irreps_in], o3.Irreps("")
        )
        gates = [None if ir.ir.l > 0 else gate for ir in irreps_in]
        self.linear_1 = Linear(
            irreps_in=irreps_in, irreps_out=self.hidden_irreps, cueq_config=cueq_config
        )
        self.non_linearity = Activation(irreps_in=self.hidden_irreps, acts=gates)
        self.linear_2 = Linear(
            irreps_in=self.hidden_irreps, irreps_out=irreps_out, cueq_config=cueq_config
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.non_linearity(self.linear_1(x))
        return self.linear_2(x)
