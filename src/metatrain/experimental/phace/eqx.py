"""JAX/Equinox port of the PhACE architecture.

This module implements the PhACE model using JAX and Equinox, enabling
JIT compilation and automatic differentiation for efficient force computation.
Only total energy output is supported (no general target handling).
"""

import math
from typing import Dict, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


# Force full float32 precision for matrix multiplications.
# JAX's default XLA GEMM kernel uses a lower-precision algorithm that
# diverges from PyTorch/numpy by ~1e-3 for typical model sizes.
jax.config.update("jax_default_matmul_precision", "float32")


# ===========================================================================
# Pure JAX utility functions
# ===========================================================================


def _cutoff_func(r, pair_cutoffs, width):
    """Smooth bump cutoff function (JAX version, uses jnp.where for safety)."""
    scaled = (r - (pair_cutoffs - width)) / width
    # Compute active-region value with a safe (clipped) argument to avoid NaN
    safe_scaled = jnp.clip(scaled, 1e-10, 1.0 - 1e-10)
    f_active = 0.5 * (1.0 + jnp.tanh(1.0 / jnp.tan(jnp.pi * safe_scaled)))
    return jnp.where(scaled <= 0.0, 1.0, jnp.where(scaled >= 1.0, 0.0, f_active))


def _spline_compute(positions, spline_positions, spline_values, spline_derivatives):
    """Hermite cubic spline interpolation (JAX version).

    :param positions: Query positions [n_pts]
    :param spline_positions: Knot positions [n_knots]
    :param spline_values: Values at knots [n_knots, n_basis]
    :param spline_derivatives: Derivatives at knots [n_knots, n_basis]
    :return: Interpolated values [n_pts, n_basis]
    """
    delta_x = spline_positions[1] - spline_positions[0]
    n = jnp.clip(
        jnp.floor(positions / delta_x).astype(jnp.int32),
        0,
        spline_positions.shape[0] - 2,
    )
    t = (positions - n * delta_x) / delta_x
    t2, t3 = t**2, t**3
    h00 = (2.0 * t3 - 3.0 * t2 + 1.0)[:, None]
    h10 = (t3 - 2.0 * t2 + t)[:, None]
    h01 = (-2.0 * t3 + 3.0 * t2)[:, None]
    h11 = (t3 - t2)[:, None]
    p_k = spline_values[n]
    p_k1 = spline_values[n + 1]
    m_k = spline_derivatives[n]
    m_k1 = spline_derivatives[n + 1]
    return h00 * p_k + h10 * delta_x * m_k + h01 * p_k1 + h11 * delta_x * m_k1


def _spherical_harmonics(xyz, F, l_max):
    """Real spherical harmonics (JAX port of SphericalHarmonicsNoSphericart).

    :param xyz: Cartesian vectors [n_pairs, 3]
    :param F: Pre-computed normalization factors [(l_max+1)*(l_max+2)//2]
    :param l_max: Maximum angular momentum
    :return: Spherical harmonics [(l_max+1)^2] stacked along axis 1
    """
    rsq = jnp.sum(xyz**2, axis=1)
    # Guard against division by zero for padded (zero-distance) pairs.
    # Their values don't matter — pair_mask zeros them before any scatter-add.
    safe_rsq = jnp.where(rsq > 0, rsq, 1.0)
    xyz_n = xyz / jnp.sqrt(safe_rsq)[:, None]
    x, y, z = xyz_n[:, 0], xyz_n[:, 1], xyz_n[:, 2]

    # Build associated Legendre polynomials Q indexed by l*(l+1)//2 + m
    Q: Dict[int, jnp.ndarray] = {}
    Q[0] = jnp.ones(xyz.shape[0], dtype=xyz.dtype)

    for l in range(1, l_max + 1):  # noqa: E741
        top = (l + 1) * (l + 2) // 2 - 1
        prev_top = l * (l + 1) // 2 - 1
        Q[top] = -(2 * l - 1) * Q[prev_top]
        second = (l + 1) * (l + 2) // 2 - 2
        Q[second] = -z * Q[top]
        for m in range(0, l - 1):
            idx = l * (l + 1) // 2 + m
            Q[idx] = (
                (2 * l - 1) * z * Q[(l - 1) * l // 2 + m]
                - (l + m - 1) * Q[(l - 2) * (l - 1) // 2 + m]
            ) / (l - m)

    # Sine and cosine of azimuthal angle (Condon-Shortley phase)
    s = [jnp.zeros_like(x), *([None] * l_max)]
    c = [jnp.ones_like(x), *([None] * l_max)]
    for m in range(1, l_max + 1):
        s[m] = x * s[m - 1] + y * c[m - 1]
        c[m] = x * c[m - 1] - y * s[m - 1]

    # Assemble real spherical harmonics Y_lm
    sqrt2 = jnp.sqrt(jnp.array(2.0, dtype=xyz.dtype))
    Y_list = []
    for l in range(l_max + 1):  # noqa: E741
        for m in range(-l, 0):
            Y_list.append(F[l * (l + 1) // 2 - m] * Q[l * (l + 1) // 2 - m] * s[-m])
        Y_list.append(F[l * (l + 1) // 2] * Q[l * (l + 1) // 2] / sqrt2)
        for m in range(1, l + 1):
            Y_list.append(F[l * (l + 1) // 2 + m] * Q[l * (l + 1) // 2 + m] * c[m])
    return jnp.stack(Y_list, axis=1)


def _get_cartesian_vectors(
    positions, cells, cell_shifts, center_indices, neighbor_indices, structure_pairs
):
    """Compute pair displacement vectors r_j - r_i + cell_shift @ cell."""
    return (
        positions[neighbor_indices]
        - positions[center_indices]
        + jnp.einsum(
            "ab,abc->ac",
            cell_shifts.astype(positions.dtype),
            cells[structure_pairs],
        )
    )


def _split_up_features(features, k_max_l):
    """Split ragged feature list into groups with equal channels per l.

    Returns split_features[l] = [features[lp][:, :, lower:upper] for lp <= l]
    where lower/upper are the channel bounds for level l.
    """
    l_max = len(k_max_l) - 1
    split_features = []
    for l in range(l_max, -1, -1):  # noqa: E741
        lower = k_max_l[l + 1] if l < l_max else 0
        upper = k_max_l[l]
        group = [features[lp][:, :, lower:upper] for lp in range(l + 1)]
        split_features = [group] + split_features
    return split_features


def _uncouple_features(features_list, U, padded_l_max):
    """Convert from coupled (spherical) to uncoupled basis.

    :param features_list: list of [n_atoms, 2l+1, n_feat] for l=0..padded_l_max
    :param U: CG matrix [(padded_l_max+1)^2, (padded_l_max+1)^2]
    :param padded_l_max: padded angular momentum
    :return: [n_atoms, padded_l_max+1, padded_l_max+1, n_feat]
    """
    n_feat = features_list[0].shape[2]
    if len(features_list) < padded_l_max + 1:
        features_list = list(features_list) + [
            jnp.zeros(
                (features_list[0].shape[0], 2 * padded_l_max + 1, n_feat),
                dtype=features_list[0].dtype,
            )
        ]
    stacked = jnp.concatenate(features_list, axis=1)  # [n, (p+1)^2, f]
    n_atoms = stacked.shape[0]
    n_sq = (padded_l_max + 1) ** 2
    stacked = stacked.swapaxes(0, 1)  # [(p+1)^2, n, f]
    uncoupled = (
        (U @ stacked.reshape(n_sq, n_atoms * n_feat))
        .reshape(n_sq, n_atoms, n_feat)
        .swapaxes(0, 1)
    )  # [n, (p+1)^2, f]
    return uncoupled.reshape(n_atoms, padded_l_max + 1, padded_l_max + 1, n_feat)


def _couple_features(features, U, padded_l_max):
    """Convert from uncoupled to coupled (spherical) basis.

    :param features: [n_atoms, padded_l_max+1, padded_l_max+1, n_feat]
    :param U: CG matrix [(padded_l_max+1)^2, (padded_l_max+1)^2]
    :param padded_l_max: padded angular momentum
    :return: list of [n_atoms, 2l+1, n_feat] for l=0..padded_l_max
    """
    n_atoms, _, _, n_feat = features.shape
    n_sq = (padded_l_max + 1) ** 2
    feat = features.reshape(n_atoms, n_sq, n_feat).swapaxes(0, 1)  # [n_sq, n, f]
    coupled = (
        (U.T @ feat.reshape(n_sq, n_atoms * n_feat))
        .reshape(n_sq, n_atoms, n_feat)
        .swapaxes(0, 1)
    )  # [n, n_sq, f]
    split_sizes = [2 * l + 1 for l in range(padded_l_max + 1)]  # noqa: E741
    result, offset = [], 0
    for sz in split_sizes:
        result.append(coupled[:, offset : offset + sz, :])
        offset += sz
    return result


def _uncouple_features_all(coupled_features, k_max_l, U_dict, l_max, padded_l_list):
    """Coupled → uncoupled basis for all l."""
    split_features = _split_up_features(coupled_features, k_max_l)
    return [
        _uncouple_features(
            split_features[l], U_dict[padded_l_list[l]], padded_l_list[l]
        )
        for l in range(l_max + 1)  # noqa: E741
    ]


def _couple_features_all(uncoupled_features, U_dict, l_max, padded_l_list):
    """Uncoupled → coupled basis for all l, concatenating across groups."""
    coupled_per = [
        _couple_features(
            uncoupled_features[l], U_dict[padded_l_list[l]], padded_l_list[l]
        )
        for l in range(l_max + 1)  # noqa: E741
    ]
    return [
        jnp.concatenate([coupled_per[lp][l] for lp in range(l, l_max + 1)], axis=-1)
        for l in range(l_max + 1)  # noqa: E741
    ]


def _tensor_product(u1_list, u2_list):
    """Tensor product in uncoupled basis.

    einsum('...ijf,...jkf->...ikf', u1, u2) / sqrt(j_dim)
    """
    return [
        jnp.einsum("...ijf,...jkf->...ikf", u1, u2) / math.sqrt(max(u1.shape[-2], 1))
        for u1, u2 in zip(u1_list, u2_list, strict=False)
    ]


def _get_adaptive_cutoffs(
    centers,
    edge_distances,
    num_neighbors_adaptive,
    num_nodes,
    max_cutoff,
    cutoff_width,
    pair_mask=None,
):
    """JAX version of the adaptive cutoff computation.

    Works without filtering: all pairs retained, masked by the resulting
    cutoff_func weights.

    :param pair_mask: optional boolean array [n_pairs] — True for real pairs,
        False for padding.  Padded pairs are excluded from the effective
        neighbour-count accumulation so they do not shift the adaptive cutoffs.
    """
    min_cutoff = 0.5
    probe_spacing = cutoff_width / 4.0
    n_probes = max(1, int((max_cutoff - min_cutoff) / probe_spacing))
    probe_cutoffs = jnp.linspace(
        min_cutoff, max_cutoff - probe_spacing, n_probes, dtype=edge_distances.dtype
    )

    # Effective neighbor count for each (probe_cutoff, center) pair
    # weights[i, j] = cutoff_func(r_j, probe_cutoffs[i], width)
    weights = _cutoff_func(
        edge_distances[None, :],  # [1, n_pairs]
        probe_cutoffs[:, None],  # [n_probes, 1]
        cutoff_width,
    )  # [n_probes, n_pairs]

    # Zero out contributions from padded pairs before accumulation
    if pair_mask is not None:
        weights = weights * pair_mask[None, :].astype(weights.dtype)

    # Sum per center atom: [n_probes, num_nodes]
    probe_num_neighbors = jax.ops.segment_sum(
        weights.T,  # [n_pairs, n_probes]
        centers,
        num_segments=num_nodes,
    ).T  # [n_probes, num_nodes] -> transpose to [num_nodes, n_probes] below

    probe_num_neighbors = probe_num_neighbors.T  # [num_nodes, n_probes]

    # Gaussian selection weights centred at num_neighbors_adaptive
    diff = probe_num_neighbors - num_neighbors_adaptive
    x_baseline = jnp.linspace(0, 1, n_probes, dtype=edge_distances.dtype)
    baseline = num_neighbors_adaptive * x_baseline**3
    diff = diff + baseline[None, :]

    if n_probes > 1:
        grad_diff = jnp.gradient(diff, axis=-1)
        width_t = jnp.abs(grad_diff).clip(1e-12)
    else:
        width_t = jnp.abs(diff) * 0.5 + 1e-12

    logw = -0.5 * (diff / width_t) ** 2
    logw = logw - logw.max(axis=-1, keepdims=True)
    selection_weights = jnp.exp(logw)
    selection_weights = selection_weights / selection_weights.sum(
        axis=-1, keepdims=True
    )

    # Weighted average of probe cutoffs → per-atom adaptive cutoff
    return jnp.dot(selection_weights, probe_cutoffs)  # [num_nodes]


# ===========================================================================
# Equinox module building blocks
# ===========================================================================


class EqxLinear(eqx.Module):
    """NTK-parametrized linear layer: y = x @ W.T / sqrt(n_in)."""

    weight: jnp.ndarray  # [n_out, n_in]
    n_feat_in: int = eqx.field(static=True)
    n_feat_out: int = eqx.field(static=True)

    def __call__(self, x):
        if self.n_feat_in == 0 or self.n_feat_out == 0:
            return x  # 0-dim pass-through
        return x @ self.weight.T * (self.n_feat_in**-0.5)


class EqxRMSNorm(eqx.Module):
    """Standard RMSNorm (equivalent to torch.nn.RMSNorm)."""

    weight: jnp.ndarray  # [d]
    eps: float = eqx.field(static=True)

    def __call__(self, x):
        # Normalise over last axis only
        rms_inv = jnp.reciprocal(
            jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        )
        return x * rms_inv * self.weight


class EqxBlockRMSNorm(eqx.Module):
    """RMSNorm for a block in the uncoupled basis [... , p+1, p+1, d]."""

    weight: jnp.ndarray  # [d]
    d: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __call__(self, x):
        if self.d == 0:
            return x
        rms_inv = jnp.reciprocal(
            jnp.sqrt(jnp.mean(x**2, axis=(-3, -2, -1), keepdims=True) + self.eps)
        )
        return x * rms_inv * self.weight


class EqxEquivariantRMSNorm(eqx.Module):
    """EquivariantRMSNorm over a list of uncoupled feature tensors."""

    rmsnorms: list  # List[EqxBlockRMSNorm]

    def __call__(self, features_list):
        return [n(f) for n, f in zip(self.rmsnorms, features_list, strict=False)]


class EqxLinearList(eqx.Module):
    """Linear layers for uncoupled features (one per l-group)."""

    linears: list  # List[EqxLinear]

    def __call__(self, features_list):
        return [lin(f) for lin, f in zip(self.linears, features_list, strict=False)]


class EqxMLPRadialBasis(eqx.Module):
    """Per-l MLP mapping radial basis → k_max_l features."""

    # weights_and_acts[l] = List[(weight_matrix, n_in, is_silu)]
    # where weight_matrix has shape [n_out, n_in] and n_in is used for NTK scaling,
    # is_silu indicates whether to apply SiLU after this linear
    mlps: list  # List[List[Tuple[EqxLinear, bool]]]
    l_max: int = eqx.field(static=True)

    def __call__(self, radial_basis_list):
        result = []
        for l, layers in enumerate(self.mlps):  # noqa: E741
            x = radial_basis_list[l]
            for linear, apply_silu in layers:
                x = linear(x)
                if apply_silu:
                    x = jax.nn.silu(x)
            result.append(x)
        return result


class EqxInvariantMessagePasser(eqx.Module):
    """First-layer message passing (invariant / scalar features)."""

    radial_basis_mlp: EqxMLPRadialBasis
    rmsnorm_weight: jnp.ndarray  # [k_max_l[0]]
    linear_in: list  # List[EqxLinear] length l_max+1
    linear_out: list  # List[EqxLinear] length l_max+1
    message_scaling: jnp.ndarray  # scalar
    l_max: int = eqx.field(static=True)

    def __call__(
        self,
        radial_basis,
        spherical_harmonics,
        centers,
        neighbors,
        n_atoms,
        initial_center_embedding,
    ):
        radial_basis = self.radial_basis_mlp(radial_basis)

        # RMSNorm over k_max_l[0] features
        center_emb = initial_center_embedding
        rms_inv = jnp.reciprocal(
            jnp.sqrt(jnp.mean(center_emb**2, axis=-1, keepdims=True) + 1e-5)
        )
        center_emb_normed = center_emb * rms_inv * self.rmsnorm_weight

        density = []
        for l in range(self.l_max + 1):  # noqa: E741
            sh_l = spherical_harmonics[l]  # [n_pairs, 2l+1]
            rb_l = radial_basis[l]  # [n_pairs, k_max_l[l]]
            lin_center = self.linear_in[l](
                center_emb_normed
            )  # [n_atoms, 1, k_max_l[l]]
            # source: sh_l[pair, m] * rb_l[pair, k] * lin_center[neighbor, 0, k]
            source = (
                sh_l[:, :, None]
                * rb_l[:, None, :]  # [n_pairs, 2l+1, k_max_l[l]]
                * lin_center[neighbors]  # [n_pairs, 1, k_max_l[l]]
            )
            # Scatter-add into per-atom density
            density_l = (
                jnp.zeros((n_atoms, sh_l.shape[1], rb_l.shape[1]), dtype=rb_l.dtype)
                .at[centers]
                .add(source)
            )
            density.append(density_l * self.message_scaling)

        for l in range(self.l_max + 1):  # noqa: E741
            density[l] = self.linear_out[l](density[l])

        density[0] = density[0] + initial_center_embedding
        return density


class EqxCGIteration(eqx.Module):
    """Single CG iteration: RMSNorm → linear → tensor-product → linear → skip."""

    linear_in: EqxLinearList
    rmsnorm: EqxEquivariantRMSNorm
    linear_out: EqxLinearList

    def __call__(self, features, U_dict, k_max_l, padded_l_list, l_max):
        features_in = features
        features = self.rmsnorm(features)
        features = self.linear_in(features)
        features = _tensor_product(features, features)
        features = self.linear_out(features)
        return [fi + fo for fi, fo in zip(features_in, features, strict=False)]


class EqxCGIterator(eqx.Module):
    """Stack of CG iterations."""

    iterations: list  # List[EqxCGIteration]

    def __call__(self, features, U_dict, k_max_l, padded_l_list, l_max):
        for it in self.iterations:
            features = it(features, U_dict, k_max_l, padded_l_list, l_max)
        return features


class EqxEquivariantMessagePasser(eqx.Module):
    """Equivariant message passing (layers 2+)."""

    radial_basis_mlp: EqxMLPRadialBasis
    linear_in: EqxLinearList
    rmsnorm: EqxEquivariantRMSNorm
    linear_out: EqxLinearList
    message_scaling: jnp.ndarray  # scalar
    l_max: int = eqx.field(static=True)
    padded_l_list: tuple = eqx.field(static=True)
    k_max_l: tuple = eqx.field(static=True)

    def __call__(
        self,
        radial_basis,
        spherical_harmonics,
        centers,
        neighbors,
        features,
        U_dict,
        n_atoms,
    ):
        n_atoms_val = n_atoms
        features_in = features
        features = self.rmsnorm(features)
        features = self.linear_in(features)

        radial_basis = self.radial_basis_mlp(radial_basis)

        # Build vector expansion in uncoupled basis
        vector_expansion = [
            spherical_harmonics[l][:, :, None] * radial_basis[l][:, None, :]
            for l in range(self.l_max + 1)  # noqa: E741
        ]
        unc_vexp = _uncouple_features_all(
            vector_expansion,
            list(self.k_max_l),
            U_dict,
            self.l_max,
            list(self.padded_l_list),
        )

        # Gather neighbor features and compute tensor product
        indexed = [f[neighbors] for f in features]
        combined = _tensor_product(unc_vexp, indexed)

        # Scatter-add pooling
        pooled = [
            (
                jnp.zeros((n_atoms_val,) + f.shape[1:], dtype=f.dtype)
                .at[centers]
                .add(f)
                * self.message_scaling
            )
            for f in combined
        ]

        features_out = self.linear_out(pooled)
        return [fi + fo for fi, fo in zip(features_in, features_out, strict=False)]


class EqxPhACE(eqx.Module):
    """Equinox port of the PhACE BaseModel (energy output only).

    Parameters are loaded from a metatrain PhACE checkpoint and include
    the composition model and scaler for full energy prediction.
    """

    # --- Precomputer (non-learnable) ---
    spline_positions: jnp.ndarray  # [n_knots]
    spline_values: jnp.ndarray  # [n_knots, n_total_basis]
    spline_derivatives: jnp.ndarray  # [n_knots, n_total_basis]
    lengthscales: jnp.ndarray  # [max_species+1]
    sph_F: jnp.ndarray  # [(l_max+1)*(l_max+2)//2]

    # CG transformation matrices
    U_list: list  # indexed by padded_l value → [p+1)^2, (p+1)^2]

    # --- BaseModel learnable parameters ---
    initial_scaling: jnp.ndarray  # scalar
    species_to_species_index: jnp.ndarray  # [max_species+1], int

    center_embedder: jnp.ndarray  # [n_species, k_max_l[0]]

    inv_message_passer: EqxInvariantMessagePasser
    cg_iterator: EqxCGIterator

    eq_message_passers: list  # List[EqxEquivariantMessagePasser]
    gen_cg_iterators: list  # List[EqxCGIterator]

    # Energy head and output layer (NTK linears + SiLU)
    head_layers: list  # List[(EqxLinear, apply_silu)]
    last_layer: EqxLinear  # [1, k_max_l[0]]

    # --- Scaler + composition (non-learnable post-processing) ---
    energy_scale: jnp.ndarray  # scalar
    composition_weights: jnp.ndarray  # [n_species]
    composition_types: jnp.ndarray  # [n_species], int (atomic numbers)

    # --- Static hyperparameters ---
    l_max: int = eqx.field(static=True)
    k_max_l: tuple = eqx.field(static=True)
    padded_l_list: tuple = eqx.field(static=True)
    n_max_l: tuple = eqx.field(static=True)
    r_cut: float = eqx.field(static=True)
    cutoff_width: float = eqx.field(static=True)
    num_neighbors_adaptive: Optional[float] = eqx.field(static=True)
    atomic_types: tuple = eqx.field(static=True)

    # -----------------------------------------------------------------------

    def _precompute(
        self,
        positions,
        cells,
        cell_shifts_full,
        center_indices_full,
        neighbor_indices_full,
        structure_pairs_full,
        cell_shifts_filt,
        center_indices_filt,
        neighbor_indices_filt,
        structure_pairs_filt,
        species,
        n_atoms,
        pair_mask_full=None,
        pair_mask_filt=None,
    ):
        """Compute spherical harmonics and radial basis for the filtered edge set.

        Uses two separate edge sets:
        - **full NL** (``*_full``): all pairs within ``r_cut``.  Only distances
          are computed from this set; they are used to derive per-atom adaptive
          cutoffs with correct JAX gradients through positions.
        - **filtered NL** (``*_filt``): pairs that survive the adaptive-cutoff
          filter (pre-selected in Python).  All heavy computation — spherical
          harmonics, radial basis, message passing — operates on this smaller
          set, keeping memory usage proportional to the ~adaptive-cutoff
          neighbour count rather than the full-cutoff one.

        :param pair_mask_full: [n_pairs_full] bool — True for real full-NL pairs
        :param pair_mask_filt: [n_pairs_filt] bool — True for real filtered pairs
        """
        # ------------------------------------------------------------------
        # 1.  Adaptive cutoffs from the FULL neighbor list (gradient-correct)
        # ------------------------------------------------------------------
        cart_vecs_full = _get_cartesian_vectors(
            positions,
            cells,
            cell_shifts_full,
            center_indices_full,
            neighbor_indices_full,
            structure_pairs_full,
        )
        r_sq_full = jnp.sum(cart_vecs_full**2, axis=-1)
        # jnp.sqrt(0) has an undefined gradient (NaN). Guard with jnp.where so
        # that padded pairs (r=0, masked by pair_mask_full) get r=1 and a
        # zero gradient rather than NaN.
        r_full = jnp.sqrt(jnp.where(r_sq_full > 0, r_sq_full, 1.0))

        if self.num_neighbors_adaptive is not None:
            atomic_cutoffs = _get_adaptive_cutoffs(
                center_indices_full,
                r_full,
                self.num_neighbors_adaptive,
                n_atoms,
                self.r_cut,
                self.cutoff_width,
                pair_mask=pair_mask_full,
            )
            # Per-pair cutoff for the FILTERED set, derived from the full-NL
            # adaptive cutoffs so that the gradient flows correctly.
            pair_cutoffs_filt = (
                atomic_cutoffs[center_indices_filt]
                + atomic_cutoffs[neighbor_indices_filt]
            ) / 2.0
        else:
            pair_cutoffs_filt = jnp.full(
                center_indices_filt.shape[0], self.r_cut, dtype=positions.dtype
            )

        # ------------------------------------------------------------------
        # 2.  Heavy computation on the FILTERED neighbor list only
        # ------------------------------------------------------------------
        cart_vecs_filt = _get_cartesian_vectors(
            positions,
            cells,
            cell_shifts_filt,
            center_indices_filt,
            neighbor_indices_filt,
            structure_pairs_filt,
        )
        r_sq_filt = jnp.sum(cart_vecs_filt**2, axis=-1)
        r_filt = jnp.sqrt(jnp.where(r_sq_filt > 0, r_sq_filt, 1.0))

        # Spherical harmonics
        sph = _spherical_harmonics(cart_vecs_filt, self.sph_F, self.l_max)
        sph = sph * jnp.sqrt(jnp.array(4.0 * jnp.pi, dtype=sph.dtype))
        split_sizes = [2 * l + 1 for l in range(self.l_max + 1)]  # noqa: E741
        spherical_harmonics = []
        offset = 0
        for sz in split_sizes:
            spherical_harmonics.append(sph[:, offset : offset + sz])
            offset += sz

        # Radial basis via spline
        center_species = species[center_indices_filt]
        neighbor_species = species[neighbor_indices_filt]
        ls_c = self.lengthscales[center_species]
        ls_n = self.lengthscales[neighbor_species]
        x = r_filt / (0.1 + jnp.exp(ls_c) + jnp.exp(ls_n))
        capped_x = jnp.where(x < 10.0, x, 5.0)
        radial_functions = jnp.where(
            x[:, None] < 10.0,
            _spline_compute(
                capped_x,
                self.spline_positions,
                self.spline_values,
                self.spline_derivatives,
            ),
            0.0,
        )
        cutoff_mul = _cutoff_func(r_filt, pair_cutoffs_filt, self.cutoff_width)
        if pair_mask_filt is not None:
            cutoff_mul = cutoff_mul * pair_mask_filt.astype(cutoff_mul.dtype)
        radial_functions = radial_functions * cutoff_mul[:, None]

        # Split radial functions by l
        radial_basis = []
        offset = 0
        for n in self.n_max_l:
            radial_basis.append(radial_functions[:, offset : offset + n])
            offset += n

        return spherical_harmonics, radial_basis

    def _get_U_dict(self, dtype):
        """Return U_dict as {padded_l: array} in the requested dtype."""
        U_dict = {}
        for padded_l, U in enumerate(self.U_list):
            if U is not None:
                U_dict[padded_l] = U.astype(dtype)
        return U_dict

    def __call__(
        self,
        positions,
        cells,
        cell_shifts_full,
        center_indices_full,
        neighbor_indices_full,
        structure_pairs_full,
        cell_shifts_filt,
        center_indices_filt,
        neighbor_indices_filt,
        structure_pairs_filt,
        species,
        n_atoms,
        pair_mask_full=None,
        pair_mask_filt=None,
    ):
        """Forward pass: compute per-atom raw energies (before scaler/composition).

        :param positions: [n_atoms, 3]
        :param cells: [n_structures, 3, 3]
        :param cell_shifts_full/filt: [n_pairs_full/filt, 3] integer cell shifts
        :param center_indices_full/filt: [n_pairs_full/filt] int
        :param neighbor_indices_full/filt: [n_pairs_full/filt] int
        :param structure_pairs_full/filt: [n_pairs_full/filt] int
        :param species: [n_atoms] int  (atomic numbers)
        :param n_atoms: Python int, padded atom count (static for JIT)
        :param pair_mask_full: [n_pairs_full] bool
        :param pair_mask_filt: [n_pairs_filt] bool
        :return: per-atom raw energies [n_atoms]
        """
        dtype = positions.dtype
        U_dict = self._get_U_dict(dtype)

        spherical_harmonics, radial_basis = self._precompute(
            positions,
            cells,
            cell_shifts_full,
            center_indices_full,
            neighbor_indices_full,
            structure_pairs_full,
            cell_shifts_filt,
            center_indices_filt,
            neighbor_indices_filt,
            structure_pairs_filt,
            species,
            n_atoms,
            pair_mask_full=pair_mask_full,
            pair_mask_filt=pair_mask_filt,
        )

        # Scale spherical harmonics
        initial_scaling = self.initial_scaling.astype(dtype)
        spherical_harmonics = [sh * initial_scaling for sh in spherical_harmonics]

        # Center embeddings
        species_idx = self.species_to_species_index[species]
        element_embedding = self.center_embedder[species_idx]  # [n_atoms, k_max_l[0]]
        element_embedding = element_embedding[:, None, :]  # [n_atoms, 1, k_max_l[0]]

        # Invariant message passing (uses filtered edges)
        features = self.inv_message_passer(
            radial_basis,
            spherical_harmonics,
            center_indices_filt,
            neighbor_indices_filt,
            n_atoms,
            element_embedding,
        )

        # Convert to uncoupled basis
        features = _uncouple_features_all(
            features,
            list(self.k_max_l),
            U_dict,
            self.l_max,
            list(self.padded_l_list),
        )

        # First CG iteration
        features = self.cg_iterator(
            features,
            U_dict,
            list(self.k_max_l),
            list(self.padded_l_list),
            self.l_max,
        )

        # Subsequent GNN layers (use filtered edges)
        for eq_mp, gen_cg in zip(
            self.eq_message_passers, self.gen_cg_iterators, strict=False
        ):
            features = eq_mp(
                radial_basis,
                spherical_harmonics,
                center_indices_filt,
                neighbor_indices_filt,
                features,
                U_dict,
                n_atoms,
            )
            features = gen_cg(
                features,
                U_dict,
                list(self.k_max_l),
                list(self.padded_l_list),
                self.l_max,
            )

        # Convert back to coupled basis
        features = _couple_features_all(
            features,
            U_dict,
            self.l_max,
            list(self.padded_l_list),
        )

        # Energy head (MLP applied to L=0 features)
        feat0 = features[0]  # [n_atoms, 1, k_max_l[0]]
        for linear, apply_silu in self.head_layers:
            feat0 = linear(feat0)
            if apply_silu:
                feat0 = jax.nn.silu(feat0)

        # Last linear layer → per-atom energy [n_atoms, 1, 1]
        energy_per_atom = self.last_layer(feat0)  # [n_atoms, 1, 1]
        return energy_per_atom[:, 0, 0]  # [n_atoms]

    def energy(
        self,
        positions,
        cells,
        cell_shifts_full,
        center_indices_full,
        neighbor_indices_full,
        structure_pairs_full,
        cell_shifts_filt,
        center_indices_filt,
        neighbor_indices_filt,
        structure_pairs_filt,
        species,
        structure_centers,
        n_atoms,
        pair_mask_full=None,
        pair_mask_filt=None,
        atom_mask=None,
    ):
        """Full energy with scaler and composition corrections.

        :param structure_centers: [n_atoms] structure index for each atom
        :param pair_mask_full: [n_pairs_full] bool — True for real full-NL pairs
        :param pair_mask_filt: [n_pairs_filt] bool — True for real filtered pairs
        :param atom_mask: [n_atoms] bool — True for real atoms, False for padding
        :return: total energy per structure (scalar for single structure)
        """
        dtype = positions.dtype
        raw = self(
            positions,
            cells,
            cell_shifts_full,
            center_indices_full,
            neighbor_indices_full,
            structure_pairs_full,
            cell_shifts_filt,
            center_indices_filt,
            neighbor_indices_filt,
            structure_pairs_filt,
            species,
            n_atoms,
            pair_mask_full=pair_mask_full,
            pair_mask_filt=pair_mask_filt,
        )
        scale = self.energy_scale.astype(dtype)
        scaled = raw * scale

        # Per-atom composition energy
        # Map atomic numbers to composition weight index
        comp_e = jnp.zeros(n_atoms, dtype=dtype)
        for i, at in enumerate(self.atomic_types):
            mask = species == at
            comp_e = comp_e + jnp.where(
                mask,
                self.composition_weights[i].astype(dtype),
                jnp.zeros(n_atoms, dtype=dtype),
            )

        total_per_atom = scaled + comp_e
        if atom_mask is not None:
            total_per_atom = total_per_atom * atom_mask.astype(dtype)
        return total_per_atom.sum()


# ===========================================================================
# Checkpoint loading
# ===========================================================================


def _t(tensor):
    """Convert a torch tensor to a numpy array."""
    return tensor.detach().cpu().numpy()


def _build_linear(w_torch):
    """Build EqxLinear from a torch weight tensor [n_out, n_in]."""
    w = _t(w_torch)
    n_out, n_in = w.shape
    return EqxLinear(weight=jnp.array(w), n_feat_in=n_in, n_feat_out=n_out)


def _build_mlp_radial_basis(sd, prefix, n_max_l, k_max_l, depth, expansion_ratio):
    """Reconstruct EqxMLPRadialBasis from state dict."""
    l_max = len(n_max_l) - 1
    mlps = []
    for l in range(l_max + 1):  # noqa: E741
        # Layer ordering in nn.Sequential:
        #   0: Linear(n_in, w*k)  [NTK]
        #   1: SiLU
        #   2: Linear(w*k, w*k)  [NTK]
        #   3: SiLU
        #   ... (depth-1 hidden layers)
        #   2*(depth-1): Linear(w*k, k) [NTK]
        layers = []
        # Input layer
        key0 = f"{prefix}.radial_mlps.{l}.0.linear_layer.weight"
        layers.append((_build_linear(sd[key0]), True))  # SiLU after
        # Hidden layers (depth-1 of them, each is Linear+SiLU)
        for d in range(depth - 1):
            key_d = f"{prefix}.radial_mlps.{l}.{2 * (d + 1)}.linear_layer.weight"
            layers.append((_build_linear(sd[key_d]), True))
        # Output layer (no SiLU)
        key_last = f"{prefix}.radial_mlps.{l}.{2 * depth}.linear_layer.weight"
        layers.append((_build_linear(sd[key_last]), False))
        mlps.append(layers)
    return EqxMLPRadialBasis(mlps=mlps, l_max=l_max)


def _build_linear_list(sd, prefix, k_max_l, expansion_ratio=1.0):
    """Reconstruct EqxLinearList from state dict."""
    l_max = len(k_max_l) - 1
    linears = []
    for l in range(l_max + 1):  # noqa: E741
        lower = k_max_l[l + 1] if l < l_max else 0
        upper = k_max_l[l]
        dim = upper - lower
        out_dim = max(0, int(dim * expansion_ratio))
        key = f"{prefix}.linears.{l}.linear_layer.weight"
        w = _t(sd[key])
        linears.append(
            EqxLinear(weight=jnp.array(w), n_feat_in=dim, n_feat_out=out_dim)
        )
    return EqxLinearList(linears=linears)


def _build_equivariant_rmsnorm(sd, prefix, k_max_l):
    """Reconstruct EqxEquivariantRMSNorm from state dict."""
    l_max = len(k_max_l) - 1
    rmsnorms = []
    for l in range(l_max + 1):  # noqa: E741
        lower = k_max_l[l + 1] if l < l_max else 0
        upper = k_max_l[l]
        dim = upper - lower
        key = f"{prefix}.rmsnorms.{l}.weight"
        w = _t(sd[key])
        rmsnorms.append(
            EqxBlockRMSNorm(
                weight=jnp.array(w),
                d=dim,
                eps=1e-6,
            )
        )
    return EqxEquivariantRMSNorm(rmsnorms=rmsnorms)


def _build_cg_iteration(sd, prefix, k_max_l, expansion_ratio):
    """Reconstruct EqxCGIteration from state dict."""
    # linear_in: k_max_l -> k_max_l * expansion_ratio
    linear_in = _build_linear_list(
        sd, f"{prefix}.linear_in", k_max_l, expansion_ratio=expansion_ratio
    )
    rmsnorm = _build_equivariant_rmsnorm(sd, f"{prefix}.rmsnorm", k_max_l)
    # linear_out operates on expanded k_max_l with 1/expansion_ratio ratio
    k_max_l_expanded = [int(k * expansion_ratio) for k in k_max_l]
    linear_out = _build_linear_list(
        sd,
        f"{prefix}.linear_out",
        k_max_l_expanded,
        expansion_ratio=1.0 / expansion_ratio,
    )
    return EqxCGIteration(linear_in=linear_in, rmsnorm=rmsnorm, linear_out=linear_out)


def _build_cg_iterator(sd, prefix, k_max_l, n_iterations, expansion_ratio):
    """Reconstruct EqxCGIterator from state dict."""
    iterations = [
        _build_cg_iteration(sd, f"{prefix}.cg_iterations.{i}", k_max_l, expansion_ratio)
        for i in range(n_iterations)
    ]
    return EqxCGIterator(iterations=iterations)


def _build_equivariant_message_passer(
    sd,
    prefix,
    n_max_l,
    k_max_l,
    radial_mlp_depth,
    mlp_expansion_ratio,
    l_max,
    padded_l_list,
):
    """Reconstruct EqxEquivariantMessagePasser from state dict."""
    radial_basis_mlp = _build_mlp_radial_basis(
        sd,
        f"{prefix}.radial_basis_mlp",
        n_max_l,
        k_max_l,
        depth=radial_mlp_depth,
        expansion_ratio=mlp_expansion_ratio,
    )
    linear_in = _build_linear_list(sd, f"{prefix}.linear_in", k_max_l)
    rmsnorm = _build_equivariant_rmsnorm(sd, f"{prefix}.rmsnorm", k_max_l)
    linear_out = _build_linear_list(sd, f"{prefix}.linear_out", k_max_l)
    message_scaling = jnp.array(_t(sd[f"{prefix}.message_scaling"]), dtype=jnp.float32)
    return EqxEquivariantMessagePasser(
        radial_basis_mlp=radial_basis_mlp,
        linear_in=linear_in,
        rmsnorm=rmsnorm,
        linear_out=linear_out,
        message_scaling=message_scaling,
        l_max=l_max,
        padded_l_list=tuple(padded_l_list),
        k_max_l=tuple(k_max_l),
    )


def load_from_checkpoint(ckpt_path: str) -> "EqxPhACE":
    """Load an EqxPhACE model from a metatrain PhACE checkpoint.

    :param ckpt_path: Path to the ``model.ckpt`` file.
    :return: Loaded :class:`EqxPhACE` model.
    """
    import metatensor.torch as mts
    import torch

    # ---- load checkpoint ----
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Handle both flat checkpoints and nested (wrapped_model_checkpoint) format
    if "wrapped_model_checkpoint" in ckpt:
        ckpt = ckpt["wrapped_model_checkpoint"]
    sd = ckpt["best_model_state_dict"]
    hypers = dict(ckpt["model_data"]["model_hypers"])  # mutable copy
    dataset_info = ckpt["model_data"]["dataset_info"]

    # Normalize hyperparameter names that were renamed between versions
    if (
        "tp_expansion_ratio" in hypers
        and "tensor_product_expansion_ratio" not in hypers
    ):
        hypers["tensor_product_expansion_ratio"] = hypers["tp_expansion_ratio"]

    # ---- rebuild BaseModel to get n_max_l, k_max_l, U_dict, padded_l_list ----
    from metatrain.experimental.phace.modules.base_model import BaseModel

    base = BaseModel(hypers, dataset_info)
    n_max_l = base.precomputer.n_max_l
    k_max_l = base.k_max_l
    l_max = base.l_max
    padded_l_list = base.padded_l_list
    U_dict_torch = base.U_dict  # {padded_l: torch.Tensor}

    # Build U_list indexed by padded_l value
    max_padded = max(padded_l_list)
    U_list = []
    for padded_l in range(max_padded + 1):
        if padded_l in U_dict_torch:
            U_list.append(jnp.array(_t(U_dict_torch[padded_l])))
        else:
            U_list.append(None)

    prefix = "fake_gradient_model.module"

    # ---- precomputer ----
    spline_positions = jnp.array(
        _t(sd[f"{prefix}.precomputer.spliner.spline_positions"])
    )
    spline_values = jnp.array(_t(sd[f"{prefix}.precomputer.spliner.spline_values"]))
    spline_derivatives = jnp.array(
        _t(sd[f"{prefix}.precomputer.spliner.spline_derivatives"])
    )
    lengthscales = jnp.array(_t(sd[f"{prefix}.precomputer.lengthscales"]))
    sph_F = jnp.array(_t(sd[f"{prefix}.precomputer.spherical_harmonics.F"]))

    # ---- species mapping ----
    species_to_species_index = jnp.array(
        _t(sd[f"{prefix}.species_to_species_index"]).astype(np.int32)
    )

    # ---- scalars ----
    initial_scaling = jnp.array(_t(sd[f"{prefix}.initial_scaling"]))

    # ---- center embedder ----
    center_embedder = jnp.array(_t(sd[f"{prefix}.center_embedder.weight"]))

    # ---- invariant message passer ----
    imp_prefix = f"{prefix}.invariant_message_passer"
    inv_radial_mlp = _build_mlp_radial_basis(
        sd,
        f"{imp_prefix}.radial_basis_mlp",
        n_max_l,
        k_max_l,
        depth=hypers["radial_basis"]["mlp_depth"],
        expansion_ratio=hypers["radial_basis"]["mlp_expansion_ratio"],
    )
    inv_rmsnorm_w = jnp.array(_t(sd[f"{imp_prefix}.rmsnorm.weight"]))
    inv_linear_in = [
        _build_linear(sd[f"{imp_prefix}.linear_in.{l}.linear_layer.weight"])
        for l in range(l_max + 1)  # noqa: E741
    ]
    inv_linear_out = [
        _build_linear(sd[f"{imp_prefix}.linear_out.{l}.linear_layer.weight"])
        for l in range(l_max + 1)  # noqa: E741
    ]
    inv_message_scaling = jnp.array(_t(sd[f"{imp_prefix}.message_scaling"]))

    inv_mp = EqxInvariantMessagePasser(
        radial_basis_mlp=inv_radial_mlp,
        rmsnorm_weight=inv_rmsnorm_w,
        linear_in=inv_linear_in,
        linear_out=inv_linear_out,
        message_scaling=inv_message_scaling,
        l_max=l_max,
    )

    # ---- first CG iterator ----
    cg_iter = _build_cg_iterator(
        sd,
        f"{prefix}.cg_iterator",
        k_max_l,
        n_iterations=hypers["num_tensor_products"],
        expansion_ratio=hypers["tensor_product_expansion_ratio"],
    )

    # ---- equivariant message passers + generalized CG iterators ----
    n_gnn = hypers["num_gnn_layers"]
    eq_mps, gen_cgs = [], []
    for i in range(n_gnn - 1):
        eq_mp = _build_equivariant_message_passer(
            sd,
            f"{prefix}.equivariant_message_passers.{i}",
            n_max_l,
            k_max_l,
            radial_mlp_depth=hypers["radial_basis"]["mlp_depth"],
            mlp_expansion_ratio=hypers["radial_basis"]["mlp_expansion_ratio"],
            l_max=l_max,
            padded_l_list=padded_l_list,
        )
        eq_mps.append(eq_mp)
        gen_cg = _build_cg_iterator(
            sd,
            f"{prefix}.generalized_cg_iterators.{i}",
            k_max_l,
            n_iterations=hypers["num_tensor_products"],
            expansion_ratio=hypers["tensor_product_expansion_ratio"],
        )
        gen_cgs.append(gen_cg)

    # ---- energy head ----
    n_head = hypers["mlp_head_num_layers"]
    head_layers = []
    if n_head == 1:
        head_layers.append(
            (_build_linear(sd[f"{prefix}.heads.energy.0.linear_layer.weight"]), True)
        )
    else:
        head_layers.append(
            (_build_linear(sd[f"{prefix}.heads.energy.0.linear_layer.weight"]), True)
        )
        for d in range(1, n_head - 1):
            head_layers.append(
                (
                    _build_linear(
                        sd[f"{prefix}.heads.energy.{2 * d}.linear_layer.weight"]
                    ),
                    True,
                )
            )
        head_layers.append(
            (
                _build_linear(
                    sd[f"{prefix}.heads.energy.{2 * (n_head - 1)}.linear_layer.weight"]
                ),
                True,
            )
        )

    last_layer = _build_linear(sd[f"{prefix}.last_layers.energy.0.linear_layer.weight"])

    # ---- scaler ----
    scaler_buf = sd["scaler.energy_scaler_buffer"]
    scaler_tm = mts.load_buffer(scaler_buf)
    energy_scale = jnp.array(float(scaler_tm.block(0).values[0, 0]))

    # ---- composition model ----
    comp_buf = sd["additive_models.0.energy_composition_buffer"]
    comp_tm = mts.load_buffer(comp_buf)
    comp_block = comp_tm.block(0)
    comp_vals = _t(comp_block.values[:, 0])  # [n_species]
    comp_types = np.array(
        [int(comp_block.samples["center_type"][i]) for i in range(len(comp_vals))]
    )
    atomic_types = tuple(int(at) for at in dataset_info.atomic_types)

    return EqxPhACE(
        spline_positions=spline_positions,
        spline_values=spline_values,
        spline_derivatives=spline_derivatives,
        lengthscales=lengthscales,
        sph_F=sph_F,
        U_list=U_list,
        initial_scaling=initial_scaling,
        species_to_species_index=species_to_species_index,
        center_embedder=center_embedder,
        inv_message_passer=inv_mp,
        cg_iterator=cg_iter,
        eq_message_passers=eq_mps,
        gen_cg_iterators=gen_cgs,
        head_layers=head_layers,
        last_layer=last_layer,
        energy_scale=energy_scale,
        composition_weights=jnp.array(comp_vals),
        composition_types=jnp.array(comp_types, dtype=jnp.int32),
        l_max=l_max,
        k_max_l=tuple(k_max_l),
        padded_l_list=tuple(padded_l_list),
        n_max_l=tuple(n_max_l),
        r_cut=float(hypers["cutoff"]),
        cutoff_width=float(hypers["cutoff_width"]),
        num_neighbors_adaptive=(
            float(hypers["num_neighbors_adaptive"])
            if hypers["num_neighbors_adaptive"] is not None
            else None
        ),
        atomic_types=atomic_types,
    )
