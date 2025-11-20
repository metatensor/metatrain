"""Utils to extract LLF from the internal head of MACE models."""
from typing import Any

from e3nn import o3
import torch
from mace.modules.blocks import LinearReadoutBlock, NonLinearReadoutBlock

from mace.tools.scatter import scatter_sum

def readout_is_linear(obj: Any):
    if isinstance(obj, torch.jit.RecursiveScriptModule):
        return obj.original_name == "LinearReadoutBlock"
    else:
        return isinstance(obj, LinearReadoutBlock)


def readout_is_nonlinear(obj: Any):
    if isinstance(obj, torch.jit.RecursiveScriptModule):
        return obj.original_name == "NonLinearReadoutBlock"
    else:
        return isinstance(obj, NonLinearReadoutBlock)

def get_llf_from_mace_readout(
    readout: torch.nn.Module,
    features: torch.Tensor,
    n_scalars: int,
    is_linear: bool,
) -> torch.Tensor:
    """Extract the learned latent features (LLF) from a MACE readout block.

    """
    if is_linear:
        ll_feats = features
    else:
        ll_feats = readout.non_linearity(readout.linear_1(features))

    return ll_feats[:, :n_scalars]

def get_llf_from_mace_model(
    readouts: list[torch.nn.Module],
    features: torch.Tensor,
    per_layer_dim: list[int],
    per_layer_nscalars: list[int],
    indices: torch.Tensor,
    num_graphs: int,
    readouts_are_linear: list[bool],
) -> torch.Tensor:
    """Extract the learned latent features (LLF) from a MACE model.

    """
    ll_feats_list = torch.split(features, per_layer_dim, dim=-1)
    ll_feats_list = [
        (
            ll_feats
            if is_linear
            else readout.non_linearity(readout.linear_1(ll_feats))
        )[:, :n_scalars]
        for ll_feats, readout, n_scalars, is_linear in zip(
            ll_feats_list,
            readouts,
            per_layer_nscalars,
            readouts_are_linear,
        )
    ]

    # Aggregate node features
    ll_feats_cat = torch.cat(ll_feats_list, dim=-1)
    ll_feats_agg = scatter_sum(
        src=ll_feats_cat, index=indices, dim=0, dim_size=num_graphs
    )

    return ll_feats_agg

class LinearReadoutLLFExtractor(torch.nn.Module):
    """Module to extract LLF from a LinearReadoutBlock."""

    def __init__(self, readout: LinearReadoutBlock, n_scalars: int):
        super().__init__()
        self.readout = readout
        self.n_scalars = n_scalars

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, : self.n_scalars]

class NonLinearReadoutLLFExtractor(torch.nn.Module):
    """Module to extract LLF from a NonLinearReadoutBlock."""

    def __init__(self, readout: NonLinearReadoutBlock, n_scalars: int):
        super().__init__()
        self.readout = readout
        self.n_scalars = n_scalars

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        ll_feats = self.readout.non_linearity(self.readout.linear_1(features))
        return ll_feats[:, : self.n_scalars]
