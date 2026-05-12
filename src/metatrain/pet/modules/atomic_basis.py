import math
from typing import List, Optional

import torch


class ZConditionedReadout(torch.nn.Module):
    """
    Species-conditioned readout module: either a single linear layer or a
    multi-layer perceptron (MLP) with SiLU activations.

    When ``z_conditioned=True``, every layer has its own per-species weight
    matrix and bias selected at runtime via ``species_idx``.  When
    ``z_conditioned=False``, a single shared set of parameters is used
    (equivalent to a standard ``torch.nn.Linear`` / MLP).  The forward
    signature is identical in both cases, so the same module container can
    hold either variant without breaking TorchScript.

    :param in_features: Number of input features per atom.
    :param out_features: Number of output features per atom.
    :param n_species: Number of atomic species (ignored when
        ``z_conditioned=False``).
    :param z_conditioned: Whether to use species-conditioned weights and
        biases for every layer.
    :param hidden_layer_widths: Widths of the hidden layers.  ``None``
        (default) gives a single linear readout
        ``in_features → out_features``.  A non-empty list gives an MLP
        ``in_features → h0 → h1 → … → out_features`` with SiLU activations
        after every hidden layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_species: int,
        z_conditioned: bool,
        hidden_layer_widths: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        self.z_conditioned = z_conditioned
        self.in_features = in_features
        self.out_features = out_features
        # When not Z-conditioned we only need a single set of weights
        # (n_species=1), which keeps the parameter layout compatible with
        # downstream code (e.g. LLPR) that accesses weights by named path.
        actual_n_species = n_species if z_conditioned else 1
        self.n_species = actual_n_species

        # Build the sequence of layer dimensions:
        #   in_features → [hidden_layer_widths...] → out_features
        if hidden_layer_widths is None:
            layer_dims: List[int] = [in_features, out_features]
        else:
            layer_dims = [in_features] + hidden_layer_widths + [out_features]

        # Allocate one batched weight matrix and bias per layer.
        # Shapes: (actual_n_species, out_dim, in_dim) and
        #         (actual_n_species, out_dim).
        weights: List[torch.nn.Parameter] = []
        biases: List[torch.nn.Parameter] = []
        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            w = torch.nn.Parameter(torch.empty(actual_n_species, out_dim, in_dim))
            b = torch.nn.Parameter(torch.empty(actual_n_species, out_dim))
            torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            bound = 1.0 / math.sqrt(in_dim) if in_dim > 0 else 0.0
            torch.nn.init.uniform_(b, -bound, bound)
            weights.append(w)
            biases.append(b)

        self.weights = torch.nn.ParameterList(weights)
        self.biases = torch.nn.ParameterList(biases)

    def forward(
        self, features: torch.Tensor, species_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        :param features: Tensor of shape ``(n_atoms, in_features)`` for node
            features, or ``(n_atoms, n_neighbours, in_features)`` for edge
            features.
        :param species_idx: Long tensor of shape ``(n_atoms,)`` containing
            the species index for each atom (ignored when
            ``z_conditioned=False``).
        :return: Tensor of shape ``(n_atoms, out_features)`` or
            ``(n_atoms, n_neighbours, out_features)``.
        """
        if self.z_conditioned:
            idx = species_idx
        else:
            # All atoms share the single set of parameters stored at index 0.
            idx = torch.zeros(
                features.shape[0], dtype=torch.long, device=features.device
            )

        # Node features arrive as 2-D; unsqueeze so every path through the
        # layers is uniformly 3-D (..., seq, dim) for the batched matmul.
        is_node = features.dim() == 2
        if is_node:
            features = features.unsqueeze(1)  # (n_atoms, 1, in_features)

        x = features
        n_layers = self.weights.__len__()  # type: ignore[attr-defined]
        for i, (W, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            W_ = W[idx]  # (n_atoms, out_dim, in_dim)
            b_ = b[idx]  # (n_atoms, out_dim)
            x = torch.matmul(x, W_.transpose(-2, -1)) + b_.unsqueeze(1)
            if i < n_layers - 1:
                x = torch.nn.functional.silu(x)

        if is_node:
            x = x.squeeze(1)
        return x


# ---------------------------------------------------------------------------
# Element-wise router
# ---------------------------------------------------------------------------


class ZElementRouter(torch.nn.Module):
    """
    Element-wise (Z-conditioned) router for mixture-of-experts.

    Maps per-atom species indices to a probability distribution over I routed
    experts.  Following Liu et al. (2025), the species is first embedded into a
    continuous latent space and then projected to I logits:

        u_i  = SiLU(Embedding(z_i))   ∈ R^M
        s_ij = softmax(W_e u_i)_j,    W_e ∈ R^{I × M}

    The ``Embedding → SiLU`` step is equivalent to a one-hidden-layer MLP on
    a one-hot encoding of the atomic number, as described in the paper.

    :param n_species: Number of distinct atomic species.
    :param num_routed_experts: Number of routed experts I.
    :param embedding_dim: Latent dimension M of the species embedding.
    """

    def __init__(
        self,
        n_species: int,
        num_routed_experts: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.species_embedding = torch.nn.Embedding(n_species, embedding_dim)
        # Routing matrix W_e — no bias, following the paper's formulation
        self.routing_matrix = torch.nn.Linear(
            embedding_dim, num_routed_experts, bias=False
        )

    def forward(self, species_idx: torch.Tensor) -> torch.Tensor:
        """
        :param species_idx: Long tensor of shape ``(n_atoms,)`` containing the
            0-based species index (into the model's ``atomic_types``) for each
            atom.
        :return: Softmax routing scores of shape
            ``(n_atoms, num_routed_experts)``.
        """
        u = torch.nn.functional.silu(
            self.species_embedding(species_idx)
        )                                            # (n_atoms, M)
        logits = self.routing_matrix(u)              # (n_atoms, I)
        return torch.softmax(logits, dim=-1)         # (n_atoms, I)


# ---------------------------------------------------------------------------
# Mixture-of-experts readout
# ---------------------------------------------------------------------------


class MoEReadout(torch.nn.Module):
    """
    Element-wise Mixture-of-Experts readout (MoE-E) from Liu et al. (2025).

    Replaces a single ``ZConditionedReadout`` with a pool of N lightweight
    expert readouts and a learnable Z-conditioned router.  Two expert types:

    * **Routed experts** (I = ``num_routed_experts``): K' of these are
      selected per atom by TopK sparse gating on the router scores.
    * **Shared experts** (N − I): always active with unit weight, capturing
      chemistry common to all species.

    Total activated experts per atom: K = K' + (N − I).

    Forward pass for atom i with features x_i and species index z_i::

        u_i   = SiLU(Embedding(z_i))               [router embedding]
        s_ij  = softmax(W_e u_i)_j,  j = 1…I      [routing scores]
        α_ij  = s_ij if j ∈ TopK'(s_i), else 0    [sparse gating weights]

        y_i = Σ_{j=1}^{I}   α_ij · f_j(x_i)       [routed, K' active]
            + Σ_{j=I+1}^{N}         f_j(x_i)       [shared, always on]

    Each expert f_j is a ``ZConditionedReadout(z_conditioned=False)``; the
    Z-dependence is expressed entirely through the routing weights α_ij.

    **TorchScript note**: all I routed experts are evaluated for every atom
    and then masked by the sparse weight tensor.  This is algebraically
    equivalent to running only the K' active experts and avoids non-uniform
    control flow, keeping the module fully scriptable.  The gradients through
    zero-weight experts are zero so inactive experts receive no update.
    Compute-sparse execution is left for future distributed work.

    :param in_features: Input feature dimension.
    :param out_features: Output feature dimension.
    :param n_species: Number of distinct atomic species.
    :param num_experts: Total number of experts N (≥ 2).
    :param num_routed_experts: Z-gated experts I (1 ≤ I ≤ N).
    :param num_topk_experts: Number of routed experts K' selected per atom
        via TopK (must satisfy 1 ≤ K' ≤ I).
    :param hidden_layer_widths: Optional hidden widths forwarded to each
        expert ``ZConditionedReadout``; ``None`` gives linear experts.
    :param embedding_dim: Latent dimension M for the species router
        (default 16).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_species: int,
        num_experts: int,
        num_routed_experts: int,
        num_topk_experts: int,
        hidden_layer_widths: Optional[List[int]] = None,
        embedding_dim: int = 16,
    ) -> None:
        super().__init__()

        num_shared_experts = num_experts - num_routed_experts

        if num_routed_experts < 1:
            raise ValueError(
                f"num_routed_experts must be >= 1, got {num_routed_experts}. "
                "Use ZConditionedReadout directly when no Z-gating is needed."
            )
        if num_shared_experts < 0:
            raise ValueError(
                f"num_routed_experts ({num_routed_experts}) exceeds "
                f"num_experts ({num_experts})."
            )
        if num_topk_experts < 1 or num_topk_experts > num_routed_experts:
            raise ValueError(
                f"num_topk_experts = {num_topk_experts}; "
                f"must satisfy 1 ≤ num_topk_experts ≤ num_routed_experts ({num_routed_experts})."
            )

        self.num_topk = num_topk_experts

        # Router: species_idx → (n_atoms, I) softmax scores
        self.router = ZElementRouter(n_species, num_routed_experts, embedding_dim)

        # All experts are standard (z_conditioned=False).
        # n_species=1 is passed but ignored by ZConditionedReadout when
        # z_conditioned=False (it always allocates actual_n_species=1).
        self.routed_experts = torch.nn.ModuleList(
            [
                ZConditionedReadout(
                    in_features,
                    out_features,
                    1,
                    z_conditioned=False,
                    hidden_layer_widths=hidden_layer_widths,
                )
                for _ in range(num_routed_experts)
            ]
        )
        self.shared_experts = torch.nn.ModuleList(
            [
                ZConditionedReadout(
                    in_features,
                    out_features,
                    1,
                    z_conditioned=False,
                    hidden_layer_widths=hidden_layer_widths,
                )
                for _ in range(num_shared_experts)
            ]
        )

    def forward(
        self, features: torch.Tensor, species_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        :param features: ``(n_atoms, in_features)`` for node features, or
            ``(n_atoms, n_neighbours, in_features)`` for edge features.
        :param species_idx: Long tensor of shape ``(n_atoms,)``.
        :return: Same leading dims as ``features``, last dim ``out_features``.
        """
        # ------------------------------------------------------------------ #
        # 1. Routing: per-atom sparse gating weights over routed experts      #
        # ------------------------------------------------------------------ #
        scores = self.router(species_idx)  # (n_atoms, I)

        # Keep only the K' largest scores; zero the rest.
        topk_scores, topk_idx = torch.topk(scores, self.num_topk, dim=-1)
        # routing_weights: (n_atoms, I), sparse — K' non-zero entries per row
        routing_weights = torch.zeros_like(scores).scatter(
            -1, topk_idx, topk_scores
        )

        # ------------------------------------------------------------------ #
        # 2. Routed experts — run all I, aggregate with sparse weights        #
        # ------------------------------------------------------------------ #
        routed_outs: List[torch.Tensor] = torch.jit.annotate(
            List[torch.Tensor], []
        )
        for expert in self.routed_experts:
            routed_outs.append(expert(features, species_idx))

        # Stack along a new expert dimension at dim=1 (n_atoms stays at dim=0):
        #   node features → (n_atoms, I, out_features)
        #   edge features → (n_atoms, I, n_neighbours, out_features)
        stacked = torch.stack(routed_outs, dim=1)

        if stacked.dim() == 3:
            # Node: routing_weights.unsqueeze(-1) → (n_atoms, I, 1)
            output = (routing_weights.unsqueeze(-1) * stacked).sum(1)
        else:
            # Edge: routing_weights.unsqueeze(-1).unsqueeze(-1) → (n_atoms, I, 1, 1)
            output = (routing_weights.unsqueeze(-1).unsqueeze(-1) * stacked).sum(1)

        # ------------------------------------------------------------------ #
        # 3. Shared experts — always active, unit weight                      #
        # ------------------------------------------------------------------ #
        for expert in self.shared_experts:
            output = output + expert(features, species_idx)

        return output
