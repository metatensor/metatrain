import math
from typing import Optional, Tuple

import torch

from .utilities import cutoff_func_bump as cutoff_func


# minimum value for the probe cutoff. this avoids getting too close
# to the central atom. in practice it could be also set to a larger value
DEFAULT_MIN_PROBE_CUTOFF = 0.5
# recommended smooth cutoff width for effective neighbor number calculation
# smaller values lead to a more "step-like" behavior, but can be
# numerically unstable. in practice this will be called with the
# same cutoff as the main cutoff function
DEFAULT_EFFECTIVE_NUM_NEIGHBORS_WIDTH = 1.0


def _n_total(
    r_per_atom: torch.Tensor,
    edge_distances: torch.Tensor,
    centers: torch.Tensor,
    num_nodes: int,
    cutoff_width: float,
    inv_max_cutoff: float,
    num_neighbors_adaptive: float,
) -> torch.Tensor:
    """Smoothed neighbor count plus cubic baseline used by the adaptive-cutoff
    root finder. Module-level (not a closure) so the surrounding function is
    torchscriptable.

    :param r_per_atom: Per-atom probe cutoff at which to evaluate ``n_total``.
    :param edge_distances: Distances between centers and their neighbors.
    :param centers: Indices of the center atom for each edge.
    :param num_nodes: Total number of center atoms.
    :param cutoff_width: Width of the smooth cutoff taper region.
    :param inv_max_cutoff: ``1 / max_cutoff``, precomputed for the baseline.
    :param num_neighbors_adaptive: Target neighbor count; sets the baseline
        amplitude.
    :return: Per-atom ``n_total(r) = sum_j cutoff_func(d_j, r, w) +
        num_neighbors_adaptive * (r / max_cutoff)**3``.
    """
    per_edge = cutoff_func(edge_distances, r_per_atom[centers], cutoff_width)
    n = torch.zeros(num_nodes, dtype=edge_distances.dtype, device=edge_distances.device)
    n.index_add_(0, centers, per_edge)
    x = r_per_atom * inv_max_cutoff
    return n + num_neighbors_adaptive * x.pow(3)


def _n_total_and_dn_dr(
    r_per_atom: torch.Tensor,
    edge_distances: torch.Tensor,
    centers: torch.Tensor,
    num_nodes: int,
    cutoff_width: float,
    inv_max_cutoff: float,
    num_neighbors_adaptive: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute n_total and its analytical r-derivative in a single pass.

    Closed form for ``cutoff_func_bump(d, r, w)`` in its active region
    ``s = (d - r + w)/w`` in (0, 1):
        f       = ½ (1 + tanh(cot(π·s)))
        df/dr   = (π / 2w) · sech²(cot(π·s)) / sin²(π·s)
    Outside the active region, f saturates to 1 (d below cutoff) or 0
    (d above cutoff) and df/dr = 0.

    :param r_per_atom: Per-atom probe cutoff at which to evaluate.
    :param edge_distances: Distances between centers and their neighbors.
    :param centers: Indices of the center atom for each edge.
    :param num_nodes: Total number of center atoms.
    :param cutoff_width: Width of the smooth cutoff taper region.
    :param inv_max_cutoff: ``1 / max_cutoff``, precomputed for the baseline.
    :param num_neighbors_adaptive: Target neighbor count; sets the baseline
        amplitude.
    :return: Tuple ``(n_total, dn_total/dr)``, both of shape ``(num_nodes,)``.
    """
    r_per_edge = r_per_atom[centers]
    scaled = (edge_distances - (r_per_edge - cutoff_width)) / cutoff_width
    active = (scaled > 0.0) & (scaled < 1.0)
    smaller = scaled <= 0.0

    # Clamp scaled into the open interval (0, 1) before evaluating the trig
    # formula. Active edges with scaled extremely close to 0 or 1 would
    # otherwise yield 0/0 = NaN in df_dr; the clamp matches the eps in
    # cutoff_func_bump so f also numerically agrees with the forward pass.
    safe = scaled.clamp(1e-6, 1.0 - 1e-6)
    s = math.pi * safe
    sin_s = torch.sin(s)
    cot_s = torch.cos(s) / sin_s
    tanh_cot = torch.tanh(cot_s)

    f_active = 0.5 * (1.0 + tanh_cot)
    f = torch.where(active, f_active, smaller.to(scaled.dtype))

    sech_sq = 1.0 - tanh_cot * tanh_cot
    df_dr = ((0.5 * math.pi / cutoff_width) * sech_sq / (sin_s * sin_s)) * active.to(
        scaled.dtype
    )

    n = torch.zeros(num_nodes, dtype=edge_distances.dtype, device=edge_distances.device)
    n.index_add_(0, centers, f)
    dn = torch.zeros(
        num_nodes, dtype=edge_distances.dtype, device=edge_distances.device
    )
    dn.index_add_(0, centers, df_dr)

    x = r_per_atom * inv_max_cutoff
    n = n + num_neighbors_adaptive * x.pow(3)
    dn = dn + 3.0 * num_neighbors_adaptive * x.pow(2) * inv_max_cutoff

    return n, dn


def get_adaptive_cutoffs_solver(
    centers: torch.Tensor,
    edge_distances: torch.Tensor,
    num_neighbors_adaptive: float,
    num_nodes: int,
    max_cutoff: float,
    cutoff_width: float = DEFAULT_EFFECTIVE_NUM_NEIGHBORS_WIDTH,
) -> torch.Tensor:
    """
    Adaptive per-atom cutoff via root-finding on the smoothed neighbor count.

    Defines
    ``n_total(r) = sum_j cutoff_func(d_j, r, cutoff_width)
                   + num_neighbors_adaptive * (r / max_cutoff)**3``.
    The cubic baseline goes from 0 at ``r = 0`` to ``num_neighbors_adaptive``
    at ``r = max_cutoff``, so ``n_total`` is monotonic non-decreasing on
    ``[0, max_cutoff]`` and crosses ``num_neighbors_adaptive`` exactly once.

    Algorithm:
      * **Forward (root finding).** Newton-bisection hybrid: each iteration
        attempts a Newton step using the analytical derivative
        ``dn_total/dr``; if the step would land outside the current bracket
        (e.g., near a "shoulder" where one bump is going from active to
        saturated and the local slope is small) it falls back to the
        bracket midpoint.
      * **Backward (gradient).** A trailing implicit-function-theorem step
        ``r_bar = r - n_residual / dn_root`` attaches gradients via the
        residual, with ``r`` and ``dn_root`` constants from the forward
        pass, so backward through this is much cheaper than differentiating
        through the iterative solver itself.

    :param centers: Indices of the center atoms.
    :param edge_distances: Distances between centers and their neighbors.
    :param num_neighbors_adaptive: Target number of neighbors per atom.
    :param num_nodes: Total number of center atoms.
    :param max_cutoff: Maximum cutoff distance to consider.
    :param cutoff_width: Width of the smooth cutoff taper region.
    :return: Adapted cutoff distances for each center atom.
    """

    # Detached view of edge_distances used for the Newton-bisection loop:
    # everything inside the loop is meant to be a "constant" from autograd's
    # point of view. Detaching means autograd never records the iteration
    # ops.
    edge_distances_d = edge_distances.detach()

    inv_max_cutoff = 1.0 / max_cutoff

    with torch.profiler.record_function("PET::adaptive_cutoff_newton"):
        # Bracket [r_lo, r_hi] with f(r_lo) <= 0 <= f(r_hi). r_lo starts at
        # 0 (where n_total = 0) and r_hi at max_cutoff (where the baseline
        # alone reaches num_neighbors_adaptive), so the root is bracketed.
        r_lo = torch.zeros(
            num_nodes,
            dtype=edge_distances.dtype,
            device=edge_distances.device,
        )
        r_hi = torch.full(
            (num_nodes,),
            max_cutoff,
            dtype=edge_distances.dtype,
            device=edge_distances.device,
        )
        r = 0.5 * r_hi
        # 10 Newton iters are typically enough to converge to float32 precision.
        for _ in range(10):
            n, dn = _n_total_and_dn_dr(
                r,
                edge_distances_d,
                centers,
                num_nodes,
                cutoff_width,
                inv_max_cutoff,
                num_neighbors_adaptive,
            )
            f = n - num_neighbors_adaptive
            below = f <= 0
            r_lo = torch.where(below, r, r_lo)
            r_hi = torch.where(below, r_hi, r)
            r_newton = r - f / dn.clamp_min(1e-6)
            inside = (r_newton >= r_lo) & (r_newton <= r_hi)
            r_mid = 0.5 * (r_lo + r_hi)
            r = torch.where(inside, r_newton, r_mid)
        _, dn_root = _n_total_and_dn_dr(
            r,
            edge_distances_d,
            centers,
            num_nodes,
            cutoff_width,
            inv_max_cutoff,
            num_neighbors_adaptive,
        )

    with torch.profiler.record_function("PET::adaptive_cutoff_ift"):
        # IFT step uses the ORIGINAL (grad-tracking) edge_distances so
        # gradient flows through n_residual back to atomic positions.
        n_residual = (
            _n_total(
                r,
                edge_distances,
                centers,
                num_nodes,
                cutoff_width,
                inv_max_cutoff,
                num_neighbors_adaptive,
            )
            - num_neighbors_adaptive
        )
        # clamp_min(1e-6) bounds the IFT correction in pathological
        # geometries where dn_root is tiny; clamp(max_cutoff/16, max_cutoff)
        # enforces a physical range. In the well-converged regime
        # n_residual is at float noise so neither clamp is active.
        # `min_cutoff_factor` is a fraction of max_cutoff used as a lower bound 
        # for the adapted cutoffs values. 
        
        min_cutoff_factor: float = 1.0 / 16.0
        adapted_atomic_cutoffs = (r - n_residual / dn_root.clamp_min(1e-6)).clamp(
            max_cutoff * min_cutoff_factor, max_cutoff
        )
    return adapted_atomic_cutoffs


def get_adaptive_cutoffs_grid(
    centers: torch.Tensor,
    edge_distances: torch.Tensor,
    num_neighbors_adaptive: float,
    num_nodes: int,
    max_cutoff: float,
    min_cutoff: float = DEFAULT_MIN_PROBE_CUTOFF,
    cutoff_width: float = DEFAULT_EFFECTIVE_NUM_NEIGHBORS_WIDTH,
    probe_spacing: Optional[float] = None,
    weight_width: Optional[float] = None,
) -> torch.Tensor:
    """
    Computes the adaptive cutoff values for each center atom using a discrete
    probe-cutoff grid and a Gaussian-weighted average of the probes whose
    smoothed neighbor counts are closest to ``num_neighbors_adaptive``.

    Superseded by :func:`get_adaptive_cutoffs_solver`, which solves the same
    selection as a Newton-bisection root find. Preserved here so existing
    checkpoints reload with their original behaviour.

    :param centers: Indices of the center atoms.
    :param edge_distances: Distances between centers and their neighbors.
    :param num_neighbors_adaptive: Target number of neighbors per atom.
    :param num_nodes: Total number of center atoms.
    :param max_cutoff: Maximum cutoff distance to consider.
    :param min_cutoff: Minimum cutoff distance to consider.
    :param cutoff_width: Width of the smooth cutoff taper region.
    :param probe_spacing: Spacing between probe cutoffs. If None, it will be
        automatically determined from the cutoff width.
    :param weight_width: Width of the cutoff selection weight function. If None, it
        will be automatically determined from the empirical neighbor counts.
    :return: Adapted cutoff distances for each center atom.
    """

    # heuristic for the grid spacing of probe cutoffs, based on a
    # the smoothness of the cutoff function
    if probe_spacing is None:
        probe_spacing = cutoff_width / 4.0
    probe_cutoffs = torch.arange(
        min_cutoff,
        max_cutoff,
        probe_spacing,
        device=edge_distances.device,
        dtype=edge_distances.dtype,
    )
    with torch.profiler.record_function("PET::get_effective_num_neighbors"):
        effective_num_neighbors = get_effective_num_neighbors(
            edge_distances,
            probe_cutoffs,
            centers,
            num_nodes,
            width=cutoff_width,
        )

    with torch.profiler.record_function("PET::get_cutoff_weights"):
        cutoffs_weights = get_gaussian_cutoff_weights(
            effective_num_neighbors,
            num_neighbors_adaptive,
            width=weight_width,
        )
    with torch.profiler.record_function("PET::calculate_adapted_cutoffs"):
        adapted_atomic_cutoffs = probe_cutoffs @ cutoffs_weights.T
    return adapted_atomic_cutoffs


def get_effective_num_neighbors(
    edge_distances: torch.Tensor,
    probe_cutoffs: torch.Tensor,
    centers: torch.Tensor,
    num_nodes: int,
    width: float = DEFAULT_EFFECTIVE_NUM_NEIGHBORS_WIDTH,
) -> torch.Tensor:
    """
    Computes the effective number of neighbors for each probe cutoff.

    :param edge_distances: Distances between centers and their neighbors.
    :param probe_cutoffs: Probe cutoff distances.
    :param centers: Indices of the center atoms.
    :param num_nodes: Total number of center atoms.
    :param width: Width of the cutoff function.
    :return: Effective number of neighbors for each center atom and probe cutoff.
    """

    weights = cutoff_func(
        edge_distances.unsqueeze(0), probe_cutoffs.unsqueeze(1), width
    )

    probe_num_neighbors = torch.zeros(
        (len(probe_cutoffs), num_nodes),
        dtype=edge_distances.dtype,
        device=edge_distances.device,
    )
    # accumulate the weights for all probe cutoffs and center atoms at once
    probe_num_neighbors.index_add_(1, centers, weights)
    probe_num_neighbors = probe_num_neighbors.T

    return probe_num_neighbors


def get_gaussian_cutoff_weights(
    effective_num_neighbors: torch.Tensor,
    num_neighbors_adaptive: float,
    width: Optional[float] = None,
) -> torch.Tensor:
    """
    Computes the weights for each probe cutoff based on
    the effective number of neighbors using Gaussian weights
    centered at the expected number of neighbors.

    :param effective_num_neighbors: Effective number of neighbors for each center atom
        and probe cutoff.
    :param num_neighbors_adaptive: Target maximum number of neighbors per atom.
    :param width: Width of the Gaussian cutoff selection function.
    :return: Weights for each probe cutoff.
    """
    # this early out prevents aggregation (taking the max) over empty lists
    if effective_num_neighbors.numel() == 0:
        return torch.empty_like(effective_num_neighbors)

    diff = effective_num_neighbors - num_neighbors_adaptive

    # adds a "baseline" corresponding to uniformly-distributed atoms
    # this has multiple "good" effects: it pushes the cutoff "out" when
    # there are few neighbors, and "in" when there are many, and it
    # stabilizes the weights with respect to variations in the neighbor
    # distribution when there are empty ranges leading to "flat"
    # neighbor count distribution
    x = torch.linspace(
        0,
        1,
        effective_num_neighbors.shape[1],
        device=effective_num_neighbors.device,
        dtype=effective_num_neighbors.dtype,
    )
    baseline = num_neighbors_adaptive * x**3

    diff = diff + baseline.unsqueeze(0)
    if width is None:
        # adaptive width from neighbor-count slope along probe axis (last dim)
        eps = 1e-12
        if diff.shape[-1] == 1:
            # Can't compute gradient from single point; use scaled diff as proxy
            width_t = diff.abs() * 0.5 + eps
        else:
            # Compute numerical gradient: centered differences for interior,
            # one-sided differences at boundaries
            (width_t,) = torch.gradient(diff, dim=-1)
            width_t = width_t.abs().clamp_min(eps)
    else:
        width_t = torch.ones_like(diff) * width

    logw = -0.5 * (diff / width_t) ** 2
    weights = torch.exp(logw - logw.max())

    # row-wise normalization of the weights
    weights_sum = weights.sum(dim=1, keepdim=True)
    weights = weights / weights_sum

    return weights
