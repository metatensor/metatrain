"""Linear readout ("last layer"), optionally with one independent linear map per
group of rows (e.g. per central-atom type)."""

import math
from typing import List, Optional

import torch


class LinearReadout(torch.nn.Module):
    """Linear readout, optionally conditioned on a per-row group index.

    With ``n_groups=None`` this is ``torch.nn.Linear`` (same parameter names, shapes
    and initialization). With ``n_groups=G``, each row uses the ``(out, in)`` weight
    and bias of its group, given by ``group_idx``.

    The gated map runs either as one GEMM against the weights of all groups followed
    by a gather ("dense"), or as one matmul per group after sorting the rows
    ("grouped"). The grouped path is used on CPU for 2-D input when
    ``n_groups * out_features > in_features``, and the dense path otherwise.

    :param in_features: Input feature dimension.
    :param out_features: Output feature dimension.
    :param n_groups: Number of groups, or ``None`` (default) for a single linear map.
    :param bias: Whether to include a learnable bias.
    """

    gated: bool
    grouped: bool
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
        self.gated = n_groups is not None
        self.n_groups = n_groups if n_groups is not None else 1
        self.in_features = in_features
        self.out_features = out_features
        self.grouped = self.gated and self.n_groups * out_features > in_features

        # The gated weight is stored as ``(n_groups, out, in)`` so that each group's
        # matrix is a contiguous slice. Each group is initialized as a
        # ``torch.nn.Linear``: ``kaiming_uniform_`` on the 3-D weight would compute
        # its fan-in as ``out_features * in_features`` instead of ``in_features``.
        weight = torch.empty(self.n_groups, out_features, in_features)
        bias_tensor = torch.empty(self.n_groups, out_features)
        for group in range(self.n_groups):
            torch.nn.init.kaiming_uniform_(weight[group], a=math.sqrt(5))
        if bias:
            bound = 1.0 / math.sqrt(in_features) if in_features > 0 else 0.0
            bias_tensor.uniform_(-bound, bound)

        if not self.gated:
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
        :param features: ``(n_rows, in_features)`` (e.g. node features) or
            ``(n_rows, n_columns, in_features)`` (e.g. edge features in NEF layout,
            or equivariant features with a component dimension).
        :param group_idx: Long tensor of shape ``(n_rows,)`` selecting the gating
            group for each row of ``features``. Ignored when ungated.
        :param present_groups: Optional sorted, unique long tensor containing every
            group in ``group_idx`` (e.g. ``torch.unique(group_idx)``). The dense path
            then only evaluates these groups. Ignored when ungated and by the
            grouped path, which skips empty groups anyway.
        :param bias_scale: Optional ``(n_rows, 1)`` factor multiplying the bias of
            each row, e.g. the total weight of the rows of ``features`` that were
            summed into it: this makes applying the readout to a weighted sum of
            features equal to the weighted sum of the readout of each feature. Only
            for 2-D ``features``.
        :return: Same leading dimensions as ``features``, with last dimension
            ``out_features``.
        """
        if not self.gated:
            bias = self.bias
            if bias_scale is None:
                out = torch.nn.functional.linear(features, self.weight, bias)
            else:
                out = torch.nn.functional.linear(features, self.weight)
                if bias is not None:
                    out = out + bias * bias_scale
        elif self.grouped and features.dim() == 2 and not features.is_cuda:
            out = self._forward_grouped(features, group_idx, bias_scale)
        else:
            out = self._forward_dense(features, group_idx, present_groups, bias_scale)
        return out

    def _forward_dense(
        self,
        features: torch.Tensor,
        group_idx: torch.Tensor,
        present_groups: Optional[torch.Tensor] = None,
        bias_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One GEMM against every group's weights, then keep the group each row needs.

        :param features: See :meth:`forward`.
        :param group_idx: See :meth:`forward`.
        :param present_groups: See :meth:`forward`.
        :param bias_scale: See :meth:`forward`.
        :return: See :meth:`forward`.
        """
        # Promote 2-D features to 3-D so that both cases are handled uniformly, and
        # undo it on the way out.
        is_2d = features.dim() == 2
        if is_2d:
            features = features.unsqueeze(1)
        n_rows = features.shape[0]
        n_columns = features.shape[1]

        weight = self.weight
        bias = self.bias
        n_groups = self.n_groups
        if present_groups is not None:
            # Keep only the present groups, and renumber the rows' groups to match.
            weight = weight.index_select(0, present_groups)
            if bias is not None:
                bias = bias.index_select(0, present_groups)
            group_idx = torch.searchsorted(present_groups, group_idx)
            n_groups = present_groups.shape[0]

        # Adding the bias before the gather below means that it picks up the matching
        # bias for free.
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

        # Keep only the group each row belongs to. ``index`` is an expanded view
        # (stride 0 along the broadcast dimensions), so it costs no real memory.
        out = out.reshape(n_rows, n_columns, n_groups, self.out_features)
        index = group_idx.reshape(n_rows, 1, 1, 1).expand(
            n_rows, n_columns, 1, self.out_features
        )
        out = out.gather(2, index).squeeze(2)

        if is_2d:
            out = out.squeeze(1)
        return out

    def _forward_grouped(
        self,
        features: torch.Tensor,
        group_idx: torch.Tensor,
        bias_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sort the rows by group and run one matmul per non-empty group.

        :param features: See :meth:`forward`, 2-D only.
        :param group_idx: See :meth:`forward`.
        :param bias_scale: See :meth:`forward`.
        :return: See :meth:`forward`.
        """
        order = torch.argsort(group_idx)
        counts: List[int] = torch.bincount(group_idx, minlength=self.n_groups).tolist()
        sorted_features = features.index_select(0, order)

        out = torch.empty(
            features.shape[0],
            self.out_features,
            dtype=features.dtype,
            device=features.device,
        )
        sorted_bias_scale: Optional[torch.Tensor] = None
        if bias_scale is not None:
            sorted_bias_scale = bias_scale.index_select(0, order)
        bias = self.bias
        start = 0
        for group in range(self.n_groups):
            count = counts[group]
            if count > 0:
                bias_group: Optional[torch.Tensor] = None
                if bias is not None:
                    bias_group = bias[group]
                if sorted_bias_scale is None:
                    out[start : start + count] = torch.nn.functional.linear(
                        sorted_features[start : start + count],
                        self.weight[group],
                        bias_group,
                    )
                else:
                    out[start : start + count] = torch.nn.functional.linear(
                        sorted_features[start : start + count], self.weight[group]
                    )
                    if bias_group is not None:
                        out[start : start + count] += (
                            bias_group * sorted_bias_scale[start : start + count]
                        )
            start += count
        return out.index_select(0, torch.argsort(order))
