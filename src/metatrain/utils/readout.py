"""Linear readout ("last layer") modules with optional group gating.

Architecture-agnostic building blocks that map per-row features to an output
dimension with a strictly *linear* map (any nonlinearity is expected to live in
whatever produces the input features). Two modules are provided:

* :class:`LinearReadout` -- a single shared linear map, or one independent linear
  map per group ("one-hot" conditioning). For atomistic models the group is
  typically the central-atom type, but the module is agnostic to what
  ``group_idx`` means.
* :class:`MoEReadout` -- a mixture of linear experts gated by routing weights
  derived from an embedding of the group index.

Both modules share the forward signature ``(features, group_idx) -> predictions``
so they are interchangeable, and both are TorchScript-compatible.
"""

import math
from typing import List, Optional

import torch


class LinearReadout(torch.nn.Module):
    """Linear readout, optionally conditioned on a per-row group index.

    With ``n_groups=None`` this is a plain linear map, equivalent to
    ``torch.nn.Linear``. With ``n_groups=G`` an independent ``(out, in)`` weight
    (and bias, if enabled) is used for each group, selected per row by
    ``group_idx``.

    Two implementations of the gated map are available; they give the same result
    and differ only in cost. Which one is used is decided **once, in the
    constructor**, from ``n_groups``, ``in_features`` and ``out_features`` alone, so
    the forward pass holds no data-dependent control flow and stays one shape-
    agnostic code path per instance.

    * **dense** (default) evaluates a single GEMM against all ``G`` weights stacked
      into one ``(G * out, in)`` matrix, then gathers the group each row needs. It
      does ``G`` times the necessary arithmetic, but as one large GEMM rather than
      ``n_rows`` tiny ones, with no sort and no device-to-host synchronisation
      (so it stays ``torch.compile`` and CUDA-graph friendly).
    * **grouped** sorts the rows by group and runs one dense matmul per group. It
      does no redundant arithmetic, but pays a sort, a ``.tolist()`` device-to-host
      sync, and a Python loop over the groups, so its per-group matmuls need to be
      big enough to amortise that.

    The redundant work in the dense path is a factor ``n_groups``, so the switch
    asks whether the widened output ``n_groups * out_features`` is still no wider
    than the input: if it is, the extra arithmetic is bounded by the cost of one
    full-width map and is worth paying to avoid the sort.

    The true crossover is not exactly at that ratio -- the grouped path also loses
    efficiency as ``n_groups`` grows, because each group's matmul gets smaller --
    but it is close. Benchmarked on CPU over ``n_groups`` in 4-100, ``n_rows`` in
    20-20000 and ``out_features`` in 4-256, this rule picks a path within 1.25x of
    the better one in every case measured, while either path used alone is off by
    up to 5x (dense, on wide outputs) or 7.5x (grouped, on features with a column
    axis). The main case it gets wrong is many groups with a very narrow output
    (e.g. 100 groups onto 8 properties), where it selects grouped and gives up
    ~1.2x against dense.

    The grouped path is restricted to 2-D features: sorting copies the whole
    feature tensor, and both that copy and the per-group matmuls scale with the
    column axis, which measured 7.5x *slower* than dense on edge-shaped
    ``(n_atoms, max_neighbours, d)`` input. 3-D input therefore always takes the
    dense path.

    :param in_features: Input feature dimension.
    :param out_features: Output feature dimension.
    :param n_groups: Number of gating groups, or ``None`` (default) for a single
        shared linear map, in which case ``group_idx`` is ignored.
    :param bias: Whether to include a learnable bias. Defaults to ``False``, which
        is the safe choice for equivariant models: a bias on a non-invariant
        output would break equivariance. Pass ``True`` for a standard affine
        linear layer.
    :param ntk_parametrization: If ``True``, use the NTK (neural tangent kernel)
        parametrization: unit-variance weights with the ``in_features ** -0.5``
        scaling applied in the forward pass rather than folded into the
        initialisation. This changes the effective per-layer learning rate, so it
        must match the convention used by the surrounding architecture (SPACE uses
        it; PET does not).
    """

    gated: bool
    grouped: bool
    n_groups: int
    in_features: int
    out_features: int
    scale: float
    bias: Optional[torch.Tensor]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_groups: Optional[int] = None,
        bias: bool = False,
        ntk_parametrization: bool = False,
    ) -> None:
        super().__init__()
        self.gated = n_groups is not None
        self.n_groups = n_groups if n_groups is not None else 1
        self.in_features = in_features
        self.out_features = out_features
        # Static choice of gated algorithm; see the class docstring. The dense path
        # does ``n_groups`` times the arithmetic, which is worth it while the
        # widened output stays no wider than the input.
        self.grouped = self.gated and self.n_groups * out_features > in_features
        # A zero-width input is possible (SPACE's feature groups can be empty), so
        # the NTK scaling falls back to 1.0 rather than dividing by zero.
        if ntk_parametrization and in_features > 0:
            self.scale = in_features ** (-0.5)
        else:
            self.scale = 1.0

        # The gated weight is stored as ``(n_groups, out, in)`` so that each group's
        # matrix is a contiguous slice, and flattened to ``(n_groups * out, in)`` in
        # the forward pass at no cost.
        weight = torch.empty(self.n_groups, out_features, in_features)
        bias_tensor = torch.empty(self.n_groups, out_features)
        if ntk_parametrization:
            weight.normal_(0.0, 1.0)
            bias_tensor.zero_()
        else:
            # Match ``torch.nn.Linear``'s default initialisation. On the 3-D weight
            # this has to be done one group at a time: ``kaiming_uniform_`` would
            # read ``fan_in = out_features * in_features`` from the full tensor
            # instead of ``in_features``, shrinking the scale by ~sqrt(out_features).
            for group in range(self.n_groups):
                torch.nn.init.kaiming_uniform_(weight[group], a=math.sqrt(5))
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

    def forward(self, features: torch.Tensor, group_idx: torch.Tensor) -> torch.Tensor:
        """
        :param features: ``(n_rows, in_features)`` (e.g. node features) or
            ``(n_rows, n_columns, in_features)`` (e.g. edge features in NEF
            layout, or equivariant features with a component dimension).
        :param group_idx: Long tensor of shape ``(n_rows,)`` selecting the gating
            group for each row of ``features``. Ignored when ungated.
        :return: Same leading dimensions as ``features``, with last dimension
            ``out_features``.
        """
        if not self.gated:
            out = torch.nn.functional.linear(features, self.weight, self.bias)
        elif self.grouped and features.dim() == 2:
            out = self._forward_grouped(features, group_idx)
        else:
            out = self._forward_dense(features, group_idx)
        if self.scale != 1.0:
            out = out * self.scale
        return out

    def _forward_dense(
        self, features: torch.Tensor, group_idx: torch.Tensor
    ) -> torch.Tensor:
        """One GEMM against every group's weights, then keep the group each row needs.

        :param features: ``(n_rows, in_features)`` or
            ``(n_rows, n_columns, in_features)``.
        :param group_idx: Long tensor of shape ``(n_rows,)``.
        :return: ``features`` with its last dimension replaced by ``out_features``.
        """
        # Promote 2-D features to 3-D so the node and edge cases are handled
        # uniformly, and undo it on the way out.
        is_2d = features.dim() == 2
        if is_2d:
            features = features.unsqueeze(1)
        n_rows = features.shape[0]
        n_columns = features.shape[1]

        # Folding the bias into the same call means the gather below picks up the
        # matching bias for free.
        bias = self.bias
        flat_bias: Optional[torch.Tensor] = None
        if bias is not None:
            flat_bias = bias.reshape(self.n_groups * self.out_features)
        out = torch.nn.functional.linear(
            features,
            self.weight.reshape(self.n_groups * self.out_features, self.in_features),
            flat_bias,
        )  # (n_rows, n_columns, n_groups * out_features)

        # Keep only the group each row belongs to. ``index`` is an expanded view
        # (stride 0 along the broadcast dimensions), so it costs no real memory.
        out = out.reshape(n_rows, n_columns, self.n_groups, self.out_features)
        index = group_idx.reshape(n_rows, 1, 1, 1).expand(
            n_rows, n_columns, 1, self.out_features
        )
        out = out.gather(2, index).squeeze(2)

        if is_2d:
            out = out.squeeze(1)
        return out

    def _forward_grouped(
        self, features: torch.Tensor, group_idx: torch.Tensor
    ) -> torch.Tensor:
        """Sort the rows by group and run one dense matmul per non-empty group.

        No redundant arithmetic, at the price of a sort, the ``.tolist()``
        device-to-host synchronisation needed to slice the sorted rows, and a Python
        loop over the groups. 2-D features only (see the class docstring).

        :param features: ``(n_rows, in_features)``.
        :param group_idx: Long tensor of shape ``(n_rows,)``.
        :return: ``(n_rows, out_features)``, back in the original row order.
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
        bias = self.bias
        start = 0
        for group in range(self.n_groups):
            count = counts[group]
            if count > 0:
                bias_group: Optional[torch.Tensor] = None
                if bias is not None:
                    bias_group = bias[group]
                out[start : start + count] = torch.nn.functional.linear(
                    sorted_features[start : start + count],
                    self.weight[group],
                    bias_group,
                )
            start += count
        return out.index_select(0, torch.argsort(order))


class MoEReadout(torch.nn.Module):
    """Mixture-of-experts linear readout, routed by a group-index embedding.

    A pool of ``num_experts`` linear experts, split into ``num_routed_experts``
    routed experts and ``num_experts - num_routed_experts`` shared experts. For
    each row the router (a small group-index embedding) produces softmax scores
    over the routed experts; the top-``num_topk_experts`` are kept (sparse gating)
    and combined with their routing weights. Shared experts are always active with
    unit weight.

    All routed experts are evaluated for every row and then masked by the sparse
    routing weights; this keeps the module free of data-dependent control flow
    (TorchScript-friendly), at the cost of doing the work of every expert. The
    sparsity is therefore an inductive bias, not a compute saving: the readout
    costs ``num_experts`` times a plain linear readout.

    :param in_features: Input feature dimension.
    :param out_features: Output feature dimension.
    :param n_groups: Number of distinct group indices (e.g. atomic types).
    :param num_experts: Total number of experts N (>= 1).
    :param num_routed_experts: Number of gated (routed) experts I (1 <= I <= N).
    :param num_topk_experts: Routed experts kept per row via TopK (1 <= K <= I).
    :param bias: Whether the expert linear maps include a learnable bias. Defaults
        to ``False`` (safe for equivariant models).
    :param embedding_dim: Latent dimension of the group-index router embedding.
    :param ntk_parametrization: Passed through to the expert
        :class:`LinearReadout` modules.
    """

    num_topk: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_groups: int,
        num_experts: int,
        num_routed_experts: int,
        num_topk_experts: int,
        bias: bool = False,
        embedding_dim: int = 16,
        ntk_parametrization: bool = False,
    ) -> None:
        super().__init__()

        num_shared_experts = num_experts - num_routed_experts
        if num_routed_experts < 1:
            raise ValueError(
                f"num_routed_experts must be >= 1, got {num_routed_experts}. "
                "Use atom_type_gating='one-hot' or false instead."
            )
        if num_shared_experts < 0:
            raise ValueError(
                f"num_routed_experts ({num_routed_experts}) exceeds "
                f"num_experts ({num_experts})."
            )
        if num_topk_experts < 1 or num_topk_experts > num_routed_experts:
            raise ValueError(
                f"num_topk_experts = {num_topk_experts}; must satisfy "
                f"1 <= num_topk_experts <= num_routed_experts "
                f"({num_routed_experts})."
            )

        self.num_topk = num_topk_experts

        # Router: group_idx -> (n_rows, num_routed_experts) softmax scores.
        self.group_embedding = torch.nn.Embedding(n_groups, embedding_dim)
        self.routing_matrix = torch.nn.Linear(
            embedding_dim, num_routed_experts, bias=False
        )

        # Experts are plain (ungated) linear readouts.
        self.routed_experts = torch.nn.ModuleList(
            [
                LinearReadout(
                    in_features,
                    out_features,
                    bias=bias,
                    ntk_parametrization=ntk_parametrization,
                )
                for _ in range(num_routed_experts)
            ]
        )
        self.shared_experts = torch.nn.ModuleList(
            [
                LinearReadout(
                    in_features,
                    out_features,
                    bias=bias,
                    ntk_parametrization=ntk_parametrization,
                )
                for _ in range(num_shared_experts)
            ]
        )

    def forward(self, features: torch.Tensor, group_idx: torch.Tensor) -> torch.Tensor:
        """
        :param features: ``(n_rows, in_features)`` or
            ``(n_rows, n_columns, in_features)``.
        :param group_idx: Long tensor of shape ``(n_rows,)`` with the group (e.g.
            central-atom type) index.
        :return: Same leading dimensions as ``features``, with last dimension
            ``out_features``.
        """
        # Routing: per-row sparse gating weights over the routed experts.
        u = torch.nn.functional.silu(self.group_embedding(group_idx))
        scores = torch.softmax(self.routing_matrix(u), dim=-1)  # (n_rows, I)
        topk_scores, topk_idx = torch.topk(scores, self.num_topk, dim=-1)
        routing_weights = torch.zeros_like(scores).scatter(-1, topk_idx, topk_scores)

        # Routed experts: evaluate all, then combine with the sparse weights.
        routed_outs: List[torch.Tensor] = torch.jit.annotate(List[torch.Tensor], [])
        for expert in self.routed_experts:
            routed_outs.append(expert(features, group_idx))
        stacked = torch.stack(routed_outs, dim=1)  # (n_rows, I, ...)

        if stacked.dim() == 3:
            # 2-D features: (n_rows, I, out) -> weights (n_rows, I, 1)
            output = (routing_weights.unsqueeze(-1) * stacked).sum(1)
        else:
            # 3-D features: (n_rows, I, n_columns, out) -> weights (n_rows, I, 1, 1)
            output = (routing_weights.unsqueeze(-1).unsqueeze(-1) * stacked).sum(1)

        # Shared experts: always active, unit weight.
        for expert in self.shared_experts:
            output = output + expert(features, group_idx)

        return output
