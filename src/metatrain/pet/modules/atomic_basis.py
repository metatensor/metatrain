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
    :param gather_chunk_size: Chunk size for the per-species weight gather in
        ``forward`` when ``z_conditioned=True``. Bounds the materialised
        ``(chunk, out_dim, in_dim)`` gather tensor instead of scaling with the
        full atom count; memory/performance only, does not affect results.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_species: int,
        z_conditioned: bool,
        hidden_layer_widths: Optional[List[int]] = None,
        gather_chunk_size: int = 128,
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
        self.gather_chunk_size = gather_chunk_size

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
        is_node = features.dim() == 2
        if is_node:
            features = features.unsqueeze(1)  # (n_atoms, 1, in_features)

        x = features
        n_layers = self.weights.__len__()  # type: ignore[attr-defined]
        if not self.z_conditioned:
            # All atoms share weights[i][0]; use F.linear to avoid
            # materialising the (n_atoms, out_dim, in_dim) gather tensor.
            # TorchScript only allows ModuleList/ParameterList indexing via a
            # literal or `enumerate`, not a `range()` loop variable.
            for i, (W, b) in enumerate(zip(self.weights, self.biases, strict=True)):
                x = torch.nn.functional.linear(x, W[0], b[0])
                if i < n_layers - 1:
                    x = torch.nn.functional.silu(x)
        else:
            idx = species_idx
            CHUNK = self.gather_chunk_size
            n_chunks = (x.shape[0] + CHUNK - 1) // CHUNK
            for i, (W, b) in enumerate(zip(self.weights, self.biases, strict=True)):
                x_out = torch.empty(
                    x.shape[0], x.shape[1], W.shape[-2], dtype=x.dtype, device=x.device
                )
                for ci in range(n_chunks):
                    s = ci * CHUNK
                    e = min(s + CHUNK, x.shape[0])
                    W_ = W[idx[s:e]]
                    b_ = b[idx[s:e]]
                    x_out[s:e] = torch.matmul(x[s:e], W_.transpose(-2, -1)) + b_.unsqueeze(1)
                x = x_out
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
        )  # (n_atoms, M)
        logits = self.routing_matrix(u)  # (n_atoms, I)
        return torch.softmax(logits, dim=-1)  # (n_atoms, I)


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
        routing_weights = torch.zeros_like(scores).scatter(-1, topk_idx, topk_scores)

        # ------------------------------------------------------------------ #
        # 2. Routed experts — run all I, aggregate with sparse weights        #
        # ------------------------------------------------------------------ #
        routed_outs: List[torch.Tensor] = torch.jit.annotate(List[torch.Tensor], [])
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


# ---------------------------------------------------------------------------
# Irrep-conditioned residual readout
# ---------------------------------------------------------------------------


class IrrepResidualReadout(torch.nn.Module):
    """
    Trunk + per-irrep nonlinear correction with species gating.

    For each output block (corresponding to one ``o3_lambda`` irrep):

    .. math::
        c_\\lambda = \\underbrace{\\text{trunk}(h_i,\\, z_i)}_{\\text{Z-readout}}
                   +\\; \\text{MLP}_\\lambda(h_i) \\odot g^z_\\lambda

    where:

    * **trunk** is a :class:`ZConditionedReadout` — the standard per-species
      linear readout, which remains the dominant prediction throughout training.
    * **MLP**_λ is a two-layer SiLU network with weights *shared across species*
      but *separate for every irrep block*.  It learns what non-linear features
      of the environment are useful for this particular angular channel.
    * **g**^z_λ is a per-species gate vector of shape ``(n_species, out_features)``,
      initialized to **zero**.  Because the gate multiplies the MLP output, the
      correction starts at exactly zero, so the first training steps are
      identical to plain Z-readout.  The gate then learns per-species how much
      — and in which output directions — to apply the nonlinear correction.

    Gradient flow at initialization
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * ``d loss / d gate`` = ``MLP(h_i) * d loss / d output`` → non-zero from
      step 1 (the randomly-initialized MLP produces non-zero activations).
    * ``d loss / d MLP params`` = ``gate * ...`` → zero initially, but the gate
      becomes non-zero after the first few steps, after which the MLP receives
      full gradient signal.

    This warm-start property means the model begins as a plain Z-readout and
    gradually develops per-irrep nonlinear corrections — useful for ablation
    because early training curves are directly comparable.

    :param in_features: Input feature dimension (``d_head``).
    :param out_features: Output dimension = ``(2λ+1) × n_properties``.
    :param n_species: Number of distinct atomic species.
    :param z_conditioned: Passed through to the trunk
        :class:`ZConditionedReadout`.  If ``True``, the trunk uses per-species
        weight matrices; if ``False``, a single shared trunk is used.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_species: int,
        z_conditioned: bool,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Trunk: standard Z-conditioned (or shared) linear readout.
        # ------------------------------------------------------------------
        self.trunk = ZConditionedReadout(
            in_features,
            out_features,
            n_species,
            z_conditioned=z_conditioned,
        )

        # ------------------------------------------------------------------
        # Per-irrep nonlinear correction MLP.
        # Hidden dimension matches in_features (d_head) — same as the node
        # head MLP used elsewhere in PET, so parameter cost is comparable.
        # ------------------------------------------------------------------
        self.correction = torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features),
            torch.nn.SiLU(),
            torch.nn.Linear(in_features, out_features),
        )

        # ------------------------------------------------------------------
        # Per-species gate on the correction output.
        # Zero initialization ensures the model starts at pure Z-readout.
        # ------------------------------------------------------------------
        self.species_gate = torch.nn.Parameter(torch.zeros(n_species, out_features))

    def forward(
        self, features: torch.Tensor, species_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        :param features: ``(n_atoms, in_features)`` for node features, or
            ``(n_atoms, n_neighbours, in_features)`` for edge features.
        :param species_idx: Long tensor of shape ``(n_atoms,)``.
        :return: Same leading dims as ``features``, last dim ``out_features``.
        """
        trunk_out = self.trunk(features, species_idx)
        correction = self.correction(features)  # (..., out_features)
        gate = self.species_gate[species_idx]  # (n_atoms, out_features)

        # For edge features (3-D input), broadcast gate over the neighbour dim.
        if features.dim() == 3:
            gate = gate.unsqueeze(1)  # (n_atoms, 1, out_features)

        return trunk_out + correction * gate


# ---------------------------------------------------------------------------
# IrrepResidual variants — Z-conditioned corrections
# ---------------------------------------------------------------------------


class IrrepResidualZOutput(torch.nn.Module):
    """
    Trunk + nonlinear correction with a Z-conditioned output layer.

    Extends :class:`IrrepResidualReadout` by replacing the scalar per-species
    gate with a full per-species weight matrix on the correction's output layer:

    .. math::
        \\text{output} = \\underbrace{W_z \\, h_i}_{\\text{trunk}}
                       + \\underbrace{(W_z^{\\prime} \\, \\text{SiLU}(U \\, h_i))}_{\\text{correction}}

    where :math:`U` is a *shared* hidden-layer weight (species-agnostic) and
    :math:`W_z^{\\prime}` is a *per-species* output-layer weight (zero-initialized).

    The hidden layer learns which nonlinear combinations of :math:`h_i` are
    universally useful; the Z-conditioned output layer decides per-species how
    to combine them.  Zero-initializing :math:`W_z^{\\prime}` gives the same
    warm-start as the scalar-gate variant: training begins at pure Z-readout.

    **Gradient flow at initialization**

    * :math:`\\partial L / \\partial W_z^{\\prime}` = ``SiLU(U h_i) ⊗ (∂L/∂output)``
      — non-zero from step 1 because the randomly-initialised :math:`U` produces
      non-zero hidden activations.

    :param in_features: Input feature dimension.
    :param out_features: Output dimension for this block.
    :param n_species: Number of distinct atomic species.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_species: int,
    ) -> None:
        super().__init__()

        # Z-conditioned trunk (dominant prediction path)
        self.trunk = ZConditionedReadout(
            in_features, out_features, n_species, z_conditioned=True
        )

        # Shared hidden layer — species-agnostic nonlinear feature extraction
        self.correction_hidden = torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features),
            torch.nn.SiLU(),
        )

        # Z-conditioned output layer — zero-initialized for warm start
        self.correction_out = ZConditionedReadout(
            in_features, out_features, n_species, z_conditioned=True
        )
        torch.nn.init.zeros_(self.correction_out.weights[0])
        torch.nn.init.zeros_(self.correction_out.biases[0])

    def forward(
        self, features: torch.Tensor, species_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        :param features: ``(n_atoms, in_features)`` or
            ``(n_atoms, n_neighbours, in_features)``.
        :param species_idx: Long tensor ``(n_atoms,)``.
        :return: Same leading dims as ``features``, last dim ``out_features``.
        """
        trunk_out = self.trunk(features, species_idx)
        hidden = self.correction_hidden(features)
        correction = self.correction_out(hidden, species_idx)
        return trunk_out + correction


class IrrepResidualFiLM(torch.nn.Module):
    """
    Trunk + nonlinear correction with FiLM species-conditioning on the inputs.

    Applies a learnable per-species affine transform (FiLM — Feature-wise Linear
    Modulation) to the features *before* the correction's hidden layer, then
    runs a shared MLP on the modulated features:

    .. math::
        x_z       &= \\gamma_z \\odot h_i + \\beta_z  \\quad\\text{(FiLM)}\\\\
        \\text{output} &= W_z h_i
                        + \\underbrace{U_\\text{out} \\, \\text{SiLU}(U_\\text{hid}\\, x_z)}_{\\text{correction}}

    where :math:`\\gamma_z` and :math:`\\beta_z` are per-species vectors of size
    ``in_features``, initialized to ones and zeros respectively (identity
    transform).  The final shared linear :math:`U_\\text{out}` is zero-initialized
    to ensure the correction is exactly zero at the start of training.

    The inductive bias differs from :class:`IrrepResidualZOutput`: species
    conditioning acts on *which features get nonlinearly mixed* (in the hidden
    layer) rather than on *how the hidden activations are projected to the
    output*.  Both effects can matter; this class isolates the former.

    Parameter cost of the FiLM layers: ``2 × n_species × in_features`` —
    independent of ``out_features``, so cheap for high-:math:`\\lambda` blocks.

    :param in_features: Input feature dimension.
    :param out_features: Output dimension for this block.
    :param n_species: Number of distinct atomic species.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_species: int,
    ) -> None:
        super().__init__()

        # Z-conditioned trunk
        self.trunk = ZConditionedReadout(
            in_features, out_features, n_species, z_conditioned=True
        )

        # FiLM parameters: per-species scale (γ) and shift (β)
        # γ = ones, β = zeros → identity at initialization
        self.film_gamma = torch.nn.Parameter(torch.ones(n_species, in_features))
        self.film_beta = torch.nn.Parameter(torch.zeros(n_species, in_features))

        # Shared correction MLP; output layer zero-initialized for warm start
        correction_hidden = torch.nn.Linear(in_features, in_features)
        correction_out = torch.nn.Linear(in_features, out_features)
        torch.nn.init.zeros_(correction_out.weight)
        torch.nn.init.zeros_(correction_out.bias)
        self.correction = torch.nn.Sequential(
            correction_hidden, torch.nn.SiLU(), correction_out
        )

    def forward(
        self, features: torch.Tensor, species_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        :param features: ``(n_atoms, in_features)`` or
            ``(n_atoms, n_neighbours, in_features)``.
        :param species_idx: Long tensor ``(n_atoms,)``.
        :return: Same leading dims as ``features``, last dim ``out_features``.
        """
        trunk_out = self.trunk(features, species_idx)

        gamma = self.film_gamma[species_idx]  # (n_atoms, in_features)
        beta = self.film_beta[species_idx]  # (n_atoms, in_features)
        if features.dim() == 3:
            gamma = gamma.unsqueeze(1)  # (n_atoms, 1, in_features)
            beta = beta.unsqueeze(1)

        modulated = gamma * features + beta
        correction = self.correction(modulated)
        return trunk_out + correction


class IrrepResidualZCorrection(torch.nn.Module):
    """
    Trunk + fully Z-conditioned correction MLP (zero-initialized output layer).

    The most expressive residual variant: *both* the hidden layer and the output
    layer of the correction carry per-species weight matrices.

    .. math::
        \\text{output} = \\underbrace{W_z \\, h_i}_{\\text{trunk}}
                       + \\underbrace{W_z^{\\prime} \\, \\text{SiLU}(V_z \\, h_i)}_{\\text{correction}}

    where :math:`V_z` and :math:`W_z^{\\prime}` are both Z-conditioned.
    Only :math:`W_z^{\\prime}` (the output layer) is zero-initialized; the
    hidden layer :math:`V_z` uses Kaiming initialization so gradients flow
    to the output weights from step 1.

    **Warm-start argument** — at initialization, :math:`W_z^{\\prime} = 0`
    so the correction is zero.  The gradient of the loss w.r.t. each entry
    of :math:`W_z^{\\prime}` is ``SiLU(V_z h_i) ⊗ (∂L/∂output)`` which is
    non-zero because :math:`V_z` is Kaiming-initialized, ensuring the output
    layer starts learning immediately.

    **Parameter cost** — ``n_species × (in² + in × out)`` for the correction,
    compared to ``in² + n_species × out`` for :class:`IrrepResidualZOutput`.
    For small ``n_species`` the difference is modest; for large datasets with
    many species it becomes significant.

    :param in_features: Input feature dimension.
    :param out_features: Output dimension for this block.
    :param n_species: Number of distinct atomic species.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_species: int,
        expansion_factor: Optional[int] = None,
    ) -> None:
        super().__init__()

        if expansion_factor is None:
            expansion_factor = 1

        # Z-conditioned trunk
        self.trunk = ZConditionedReadout(
            in_features, out_features, n_species, z_conditioned=True
        )

        # Fully Z-conditioned two-layer correction MLP
        # hidden_layer_widths=[in_features] → weights[0] is hidden, weights[1] is output
        self.correction = ZConditionedReadout(
            in_features,
            out_features,
            n_species,
            z_conditioned=True,
            hidden_layer_widths=[in_features * expansion_factor],
        )
        # Zero-init output layer only — hidden layer retains Kaiming init
        torch.nn.init.zeros_(self.correction.weights[1])
        torch.nn.init.zeros_(self.correction.biases[1])

    def forward(
        self, features: torch.Tensor, species_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        :param features: ``(n_atoms, in_features)`` or
            ``(n_atoms, n_neighbours, in_features)``.
        :param species_idx: Long tensor ``(n_atoms,)``.
        :return: Same leading dims as ``features``, last dim ``out_features``.
        """
        return self.trunk(features, species_idx) + self.correction(
            features, species_idx
        )


# ---------------------------------------------------------------------------
# IrrepResidualZCorrectionDeep — depth-parameterised Z-conditioned correction
# ---------------------------------------------------------------------------


class IrrepResidualZCorrectionDeep(torch.nn.Module):
    """
    Trunk + deep fully Z-conditioned correction tower (zero-initialized output).

    Generalises :class:`IrrepResidualZCorrection` to ``num_correction_layers``
    hidden layers, each carrying per-species weight matrices.  The correction
    tower is a :class:`ZConditionedReadout` with
    ``hidden_layer_widths=[in_features] * num_correction_layers``:

    .. math::
        h^{(0)} &= h_i \\\\
        h^{(k)} &= \\text{SiLU}\\!\\left(V_z^{(k)}\\, h^{(k-1)}\\right),
                   \\quad k = 1,\\ldots,K \\\\
        \\text{output} &= W_z\\, h_i
                        + \\underbrace{W_z^{\\prime}\\, h^{(K)}}_{=\\,0\\text{ at init}}

    Only the output weight :math:`W_z^{\\prime}` is zero-initialized; all
    hidden layers :math:`V_z^{(k)}` use Kaiming initialization, so gradients
    reach :math:`W_z^{\\prime}` from step 1 (the hidden activations are
    non-zero).  Training therefore starts at pure Z-readout and the correction
    grows in as it identifies what the linear trunk cannot capture.

    **Why depth helps** — a :math:`K`-hidden-layer network can compose
    :math:`K` levels of nonlinearity, accessing increasingly complex cross-term
    interactions between PET feature dimensions.  As the backbone grows
    (larger ``d_pet``, more GNN layers), the features :math:`h_i` become
    richer; additional correction depth extracts correspondingly higher-order
    structure from that richer representation.  The correction's parameter cost
    scales as :math:`n_Z \\times K \\times d^2`, so it grows naturally with
    model width without any manual tuning.

    **Recommended sweep** — ``num_correction_layers`` ∈ {1, 2, 3} at each
    model size; a monotone improvement with depth is a reliable signal that
    the correction is not yet saturated.

    :param in_features: Input feature dimension (``d_head``).
    :param out_features: Output dimension for this block.
    :param n_species: Number of distinct atomic species.
    :param num_correction_layers: Number of hidden layers ``K`` in the
        correction tower.  ``1`` recovers :class:`IrrepResidualZCorrection`
        exactly.  Default ``2``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_species: int,
        num_correction_layers: int = 2,
    ) -> None:
        super().__init__()

        if num_correction_layers < 1:
            raise ValueError(
                f"num_correction_layers must be >= 1, got {num_correction_layers}. "
                "Use IrrepResidualZCorrection (or ZConditioned) for shallower readouts."
            )

        # Z-conditioned linear trunk — the dominant, warm-started prediction path
        self.trunk = ZConditionedReadout(
            in_features, out_features, n_species, z_conditioned=True
        )

        # Deep Z-conditioned correction tower.
        # hidden_layer_widths=[in_features] * K gives K hidden layers, each
        # of dimension in_features.  The final (output) layer maps to out_features.
        # Layer index layout in self.correction.weights:
        #   [0 .. K-1] → hidden layers  (Kaiming init, kept)
        #   [K]        → output layer   (zero-init for warm start)
        self.correction = ZConditionedReadout(
            in_features,
            out_features,
            n_species,
            z_conditioned=True,
            hidden_layer_widths=[in_features] * num_correction_layers,
        )
        # Zero-init output layer only — hidden layers keep Kaiming init so
        # the output layer receives non-zero gradients from step 1.
        torch.nn.init.zeros_(self.correction.weights[num_correction_layers])
        torch.nn.init.zeros_(self.correction.biases[num_correction_layers])

    def forward(
        self, features: torch.Tensor, species_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        :param features: ``(n_atoms, in_features)`` or
            ``(n_atoms, n_neighbours, in_features)``.
        :param species_idx: Long tensor ``(n_atoms,)``.
        :return: Same leading dims as ``features``, last dim ``out_features``.
        """
        return self.trunk(features, species_idx) + self.correction(
            features, species_idx
        )


# ---------------------------------------------------------------------------
# IrrepThenZConditioned  — per-irrep MLP → Z-conditioned linear/MLP
# ---------------------------------------------------------------------------


class IrrepThenZConditioned(torch.nn.Module):
    """
    Per-irrep shared MLP followed by a Z-conditioned (or shared) linear readout.

    Maps last-layer features through two sequential stages:

    1. **Irrep MLP** ``d → d'(α)`` — a single SiLU-activated linear layer whose
       weights are *shared across species* but *unique per irrep block*.  This
       extracts features that are specifically useful for angular channel α,
       independently of which atomic species is at the centre.

    2. **Z-conditioned readout** ``d'(α) → q(α, Z)`` — a
       :class:`ZConditionedReadout` (optionally with further hidden layers)
       that separates species-specific behaviour from the irrep-specific
       features.

    The separation of roles mirrors the factorisation

    .. math::
        q(\\alpha, Z) = f_Z\\!\\left(g_\\alpha(h_i)\\right)

    where :math:`g_\\alpha` is species-agnostic (learns *what* is informative
    for irrep :math:`\\alpha`) and :math:`f_Z` is irrep-agnostic (learns *how*
    species :math:`Z` modulates those features).

    :param in_features: Input feature dimension ``d`` (``d_head``).
    :param out_features: Total output dimension for this block.
    :param n_species: Number of distinct atomic species.
    :param d_irrep: Hidden dimension ``d'(α)`` produced by the irrep MLP.
        Defaults to ``in_features``.
    :param z_conditioned: Whether the final readout uses per-species weights.
    :param hidden_layer_widths: Optional additional hidden layers inside the
        Z-conditioned readout (forwarded to :class:`ZConditionedReadout`).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_species: int,
        z_conditioned: bool = True,
        hidden_layer_widths: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        # Stage 1: per-irrep shared feature extraction
        self.irrep_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features),
            torch.nn.SiLU(),
        )

        # Stage 2: Z-conditioned (or shared) linear/MLP readout
        self.readout = ZConditionedReadout(
            in_features,
            out_features,
            n_species,
            z_conditioned=z_conditioned,
            hidden_layer_widths=hidden_layer_widths,
        )

    def forward(
        self, features: torch.Tensor, species_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        :param features: ``(n_atoms, in_features)`` or
            ``(n_atoms, n_neighbours, in_features)``.
        :param species_idx: Long tensor ``(n_atoms,)``.
        :return: Same leading dims as ``features``, last dim ``out_features``.
        """
        return self.readout(self.irrep_mlp(features), species_idx)


# ---------------------------------------------------------------------------
# IrrepThenMoE  — per-irrep MLP → MoE readout
# ---------------------------------------------------------------------------


class IrrepThenMoE(torch.nn.Module):
    """
    Per-irrep shared MLP followed by a Mixture-of-Experts readout.

    Identical in structure to :class:`IrrepThenZConditioned` but replaces the
    Z-conditioned linear with a :class:`MoEReadout`:

    .. math::
        q(\\alpha, Z) = \\text{MoE}_\\alpha\\!\\left(g_\\alpha(h_i),\\, z_i\\right)

    This combines per-irrep feature extraction (shared MLP, stage 1) with
    soft Z-gated expert selection (stage 2), giving the model both angular
    specialisation and species-dependent routing.

    :param in_features: Input feature dimension ``d``.
    :param out_features: Total output dimension for this block.
    :param n_species: Number of distinct atomic species.
    :param d_irrep: Hidden dimension produced by the irrep MLP.
    :param num_experts: Total number of experts N.
    :param num_routed_experts: Z-gated routed experts I (1 ≤ I ≤ N).
    :param num_topk_experts: TopK experts K' selected per atom (1 ≤ K' ≤ I).
    :param embedding_dim: Species embedding dimension for the MoE router.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_species: int,
        d_irrep: int,
        num_experts: int,
        num_routed_experts: int,
        num_topk_experts: int,
        embedding_dim: int = 16,
    ) -> None:
        super().__init__()

        # Stage 1: per-irrep shared feature extraction
        self.irrep_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features, d_irrep),
            torch.nn.SiLU(),
        )

        # Stage 2: MoE readout
        self.readout = MoEReadout(
            d_irrep,
            out_features,
            n_species,
            num_experts,
            num_routed_experts,
            num_topk_experts,
            embedding_dim=embedding_dim,
        )

    def forward(
        self, features: torch.Tensor, species_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        :param features: ``(n_atoms, in_features)`` or
            ``(n_atoms, n_neighbours, in_features)``.
        :param species_idx: Long tensor ``(n_atoms,)``.
        :return: Same leading dims as ``features``, last dim ``out_features``.
        """
        return self.readout(self.irrep_mlp(features), species_idx)
