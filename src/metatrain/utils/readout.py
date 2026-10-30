"""Linear readout ("last layer"), optionally with one independent linear map per
group of rows (e.g. per central-atom type)."""

import math
from typing import Optional

import torch


class LinearReadout(torch.nn.Module):
    """Linear readout, optionally conditioned on a per-row group index.

    With ``n_groups=None`` this is ``torch.nn.Linear`` (same parameter names, shapes
    and initialization). With ``n_groups=G``, each row uses the ``(out, in)`` weight
    and bias of its group, given by ``group_idx``. This runs as one GEMM against the
    weights of all groups, followed by a gather of the group each row needs.

    :param in_features: Input feature dimension.
    :param out_features: Output feature dimension.
    :param n_groups: Number of groups, or ``None`` (default) for a single linear map.
    :param bias: Whether to include a learnable bias.
    """

    conditioned: bool
    n_groups: int
    in_features: int
    out_features: int
    bias: Optional[torch.Tensor]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_groups: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conditioned = n_groups is not None
        self.n_groups = n_groups if n_groups is not None else 1
        self.in_features = in_features
        self.out_features = out_features

        # each group is initialized as a ``torch.nn.Linear``: ``kaiming_uniform_`` on
        # the 3-D weight would use ``out_features * in_features`` as the fan-in
        weight = torch.empty(self.n_groups, out_features, in_features)
        bias_tensor = torch.empty(self.n_groups, out_features)
        for group in range(self.n_groups):
            torch.nn.init.kaiming_uniform_(weight[group], a=math.sqrt(5))
        if bias:
            bound = 1.0 / math.sqrt(in_features)
            bias_tensor.uniform_(-bound, bound)

        if not self.conditioned:
            weight = weight.squeeze(0)
            bias_tensor = bias_tensor.squeeze(0)
        self.weight = torch.nn.Parameter(weight)
        if bias:
            self.bias = torch.nn.Parameter(bias_tensor)
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        features: torch.Tensor,
        group_idx: torch.Tensor,
        present_groups: Optional[torch.Tensor] = None,
        bias_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply the readout of the group of each row.

        :param features: ``(n_rows, in_features)`` (e.g. node features) or
            ``(n_rows, n_columns, in_features)`` (e.g. edge features in NEF layout,
            or equivariant features with a component dimension).
        :param group_idx: Long tensor of shape ``(n_rows,)`` selecting the conditioning
            group for each row of ``features``. Ignored when unconditioned.
        :param present_groups: Optional sorted, unique long tensor containing every
            group in ``group_idx`` (e.g. ``torch.unique(group_idx)``), so that only
            these groups are evaluated. Ignored when unconditioned.
        :param bias_scale: Optional ``(n_rows, 1)`` factor multiplying the bias of
            each row, e.g. the total weight of the rows of ``features`` that were
            summed into it: this makes applying the readout to a weighted sum of
            features equal to the weighted sum of the readout of each feature. Only
            for 2-D ``features``.
        :return: Same leading dimensions as ``features``, with last dimension
            ``out_features``.
        """
        bias = self.bias
        if not self.conditioned:
            if bias_scale is None:
                return torch.nn.functional.linear(features, self.weight, bias)
            out = torch.nn.functional.linear(features, self.weight)
            if bias is not None:
                out = out + bias * bias_scale
            return out

        # 2-D features are handled as 3-D ones with a single column
        is_2d = features.dim() == 2
        if is_2d:
            features = features.unsqueeze(1)
        n_rows = features.shape[0]
        n_columns = features.shape[1]

        weight = self.weight
        n_groups = self.n_groups
        if present_groups is not None:
            # Keep only the present groups, and renumber the rows' groups to match.
            weight = weight.index_select(0, present_groups)
            if bias is not None:
                bias = bias.index_select(0, present_groups)
            group_idx = torch.searchsorted(present_groups, group_idx)
            n_groups = present_groups.shape[0]

        # one GEMM against the weights of all groups, keeping below the group of each
        # row
        flat_bias: Optional[torch.Tensor] = None
        if bias is not None:
            flat_bias = bias.reshape(n_groups * self.out_features)
        flat_weight = weight.reshape(n_groups * self.out_features, self.in_features)
        if bias_scale is None:
            out = torch.nn.functional.linear(features, flat_weight, flat_bias)
        else:
            out = torch.nn.functional.linear(features, flat_weight)
            if flat_bias is not None:
                out = out + flat_bias * bias_scale.unsqueeze(1)
        # out: (n_rows, n_columns, n_groups * out_features)

        out = out.reshape(n_rows, n_columns, n_groups, self.out_features)
        index = group_idx.reshape(n_rows, 1, 1, 1).expand(
            n_rows, n_columns, 1, self.out_features
        )
        out = out.gather(2, index).squeeze(2)

        if is_2d:
            out = out.squeeze(1)
        return out
