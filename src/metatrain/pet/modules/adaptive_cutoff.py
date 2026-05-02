from typing import Optional

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
# number of Illinois regula-falsi iterations used to bracket the root.
# Illinois converges superlinearly (asymptotic order ~1.618) so 15 steps
# tighten the bracket far below float32 precision while still leaving
# (n_hi - n_lo) numerically meaningful for the final secant step.
N_ROOT_ITERS = 15


def get_adaptive_cutoffs(
    centers: torch.Tensor,
    edge_distances: torch.Tensor,
    num_neighbors_adaptive: float,
    num_nodes: int,
    max_cutoff: float,
    min_cutoff: float = DEFAULT_MIN_PROBE_CUTOFF,
    cutoff_width: float = DEFAULT_EFFECTIVE_NUM_NEIGHBORS_WIDTH,
    probe_spacing: Optional[float] = None,
    weight_width: Optional[float] = None,
    nl_cutoff: Optional[float] = None,
) -> torch.Tensor:
    """
    Computes the adaptive cutoff values for each center atom by solving
    ``n_total(r) = num_neighbors_adaptive`` per atom via the Illinois
    variant of regula falsi.

    The smoothed neighbor count
    ``n(r) = sum_j cutoff_func(d_j, r, cutoff_width) * nl_taper_j``
    is monotonic non-decreasing in ``r``. We add a cubic baseline
    ``num_neighbors_adaptive * x(r)**3`` with
    ``x(r) = (r - min_cutoff) / (max_cutoff - min_cutoff)`` so that
    ``n_total(min_cutoff) = 0`` and ``n_total(max_cutoff) >= num_neighbors_adaptive``,
    guaranteeing a unique root in the interval. Illinois regula falsi
    finds the root inside ``torch.no_grad`` (faster than bisection on
    smooth monotonic functions, and avoids the stagnation pathology of
    plain regula falsi by halving the residual at any endpoint that has
    been "stuck" for two consecutive iterations). Because the bracket
    converges to machine precision, a final secant step would have a
    numerically degenerate slope; gradients are instead attached via a
    Newton step in which ``dn_total/dr`` is evaluated once at the root
    with ``edge_distances`` detached. This yields exactly the implicit
    function theorem result
    ``dr_bar/d edge_distances = -(d n_total/d edge_distances)|_{r_bar}
                                 / (d n_total/d r)|_{r_bar}``.

    :param centers: Indices of the center atoms.
    :param edge_distances: Distances between centers and their neighbors.
    :param num_neighbors_adaptive: Target number of neighbors per atom.
    :param num_nodes: Total number of center atoms.
    :param max_cutoff: Maximum cutoff distance to consider.
    :param min_cutoff: Minimum cutoff distance to consider.
    :param cutoff_width: Width of the smooth cutoff taper region.
    :param probe_spacing: Accepted for API compatibility with the legacy
        grid-based implementation; ignored by the bisection algorithm.
    :param weight_width: Accepted for API compatibility with the legacy
        grid-based implementation; ignored by the bisection algorithm.
    :param nl_cutoff: Optional smooth-taper distance applied to each edge's
        contribution to the neighbor count. Use this when the host
        neighbor list is tighter than ``max_cutoff`` so that an edge whose
        distance crosses ``nl_cutoff`` in MD has its contribution go to
        zero continuously.
    :return: Adapted cutoff distances for each center atom.
    """
    del probe_spacing, weight_width  # accepted for API compat, unused

    # per-edge taper at host neighbor-list cutoff, independent of r
    if nl_cutoff is not None:
        nl_cutoff_t = edge_distances.new_full((), nl_cutoff)
        edge_taper = cutoff_func(edge_distances, nl_cutoff_t, cutoff_width)
    else:
        edge_taper = None

    inv_range = 1.0 / (max_cutoff - min_cutoff)

    def n_total(r_per_atom: torch.Tensor) -> torch.Tensor:
        per_edge = cutoff_func(edge_distances, r_per_atom[centers], cutoff_width)
        if edge_taper is not None:
            per_edge = per_edge * edge_taper
        n = torch.zeros(
            num_nodes, dtype=edge_distances.dtype, device=edge_distances.device
        )
        n.index_add_(0, centers, per_edge)
        # r_per_atom is bounded in [min_cutoff, max_cutoff] by Illinois
        # construction, so x naturally lies in [0, 1] without an explicit
        # clamp; avoiding clamp keeps the baseline gradient nonzero at the
        # boundaries (matters for atoms whose adaptive cutoff lands at
        # max_cutoff, e.g. when no neighbor contributes within nl_cutoff).
        x = (r_per_atom - min_cutoff) * inv_range
        return n + num_neighbors_adaptive * x.pow(3)

    with torch.profiler.record_function("PET::adaptive_cutoff_illinois"):
        with torch.no_grad():
            r_lo = torch.full(
                (num_nodes,),
                min_cutoff,
                dtype=edge_distances.dtype,
                device=edge_distances.device,
            )
            r_hi = torch.full(
                (num_nodes,),
                max_cutoff,
                dtype=edge_distances.dtype,
                device=edge_distances.device,
            )
            # residuals f(r) = n_total(r) - num_neighbors_adaptive.
            # by construction f_lo <= 0 and f_hi >= 0 throughout the loop.
            f_lo = n_total(r_lo) - num_neighbors_adaptive
            f_hi = n_total(r_hi) - num_neighbors_adaptive
            last_below = torch.zeros_like(r_lo, dtype=torch.bool)
            last_above = torch.zeros_like(r_lo, dtype=torch.bool)
            for _ in range(N_ROOT_ITERS):
                # regula falsi step using current (possibly Illinois-halved) residuals
                denom = (f_hi - f_lo).clamp_min(
                    torch.finfo(edge_distances.dtype).tiny
                )
                r_new = r_lo - f_lo * (r_hi - r_lo) / denom
                f_new = n_total(r_new) - num_neighbors_adaptive
                # use <= so that an exact-zero residual collapses the bracket
                # (e.g. the degenerate case where n_total(max_cutoff) is
                # exactly num_neighbors_adaptive because no neighbor contributes)
                below = f_new <= 0

                # Illinois: if the same endpoint is replaced two iters in a row,
                # halve the residual at the OPPOSITE (stuck) endpoint to break
                # stagnation. The halving applies BEFORE the bracket update,
                # so it only affects the stuck side; the side that just moves
                # gets its residual overwritten by f_new below.
                same_below = below & last_below
                same_above = (~below) & last_above
                f_hi = torch.where(same_below, 0.5 * f_hi, f_hi)
                f_lo = torch.where(same_above, 0.5 * f_lo, f_lo)

                # update bracket
                r_lo = torch.where(below, r_new, r_lo)
                f_lo = torch.where(below, f_new, f_lo)
                r_hi = torch.where(below, r_hi, r_new)
                f_hi = torch.where(below, f_hi, f_new)

                last_below = below
                last_above = ~below
            #for _ in range(N_BISECT_ITERS):
            #    r_mid = 0.5 * (r_lo + r_hi)
            #    below = n_total(r_mid) < num_neighbors_adaptive
            #    r_lo = torch.where(below, r_mid, r_lo)
            #    r_hi = torch.where(below, r_hi, r_mid)

    with torch.profiler.record_function("PET::adaptive_cutoff_ift"):
        # Implicit-function-theorem-based gradient. The bracket from Illinois
        # has converged to machine precision so a final secant step can be
        # numerically degenerate; instead we attach gradients via a Newton
        # step in which the local slope dn_total/dr is treated as a constant
        # (computed with edge_distances detached). Concretely we return
        #
        #     r_bar = r_root - (n_total(r_root) - n_bar) / dn_dr
        #
        # with ``r_root`` detached and ``dn_dr`` detached. The forward value
        # is r_root (the residual is ~0 at convergence). The backward gives
        # ``dr_bar/d edge_distances = -(d n_total / d edge_distances)|_{r_root}
        #                              / (d n_total / d r)|_{r_root}``,
        # which is the implicit function theorem result.
        r_root = r_lo.detach()

        # dn_total/dr at r_root, holding edge_distances constant
        with torch.enable_grad():
            r_for_dr = r_root.clone().requires_grad_(True)
            ed_const = edge_distances.detach()
            per_edge_const = cutoff_func(
                ed_const, r_for_dr[centers], cutoff_width
            )
            if nl_cutoff is not None:
                nl_cutoff_t_const = ed_const.new_full((), nl_cutoff)
                per_edge_const = per_edge_const * cutoff_func(
                    ed_const, nl_cutoff_t_const, cutoff_width
                )
            n_const = torch.zeros(
                num_nodes, dtype=ed_const.dtype, device=ed_const.device
            )
            n_const.index_add_(0, centers, per_edge_const)
            x_const = ((r_for_dr - min_cutoff) * inv_range).clamp(0.0, 1.0)
            n_eval = n_const + num_neighbors_adaptive * x_const.pow(3)
            (dn_dr,) = torch.autograd.grad(n_eval.sum(), r_for_dr)
            dn_dr = dn_dr.detach()

        # residual at r_root, with edge_distances carrying gradients
        n_residual = n_total(r_root) - num_neighbors_adaptive
        adapted_atomic_cutoffs = r_root - n_residual / dn_dr
    return adapted_atomic_cutoffs


def get_adaptive_cutoffs_old(
    centers: torch.Tensor,
    edge_distances: torch.Tensor,
    num_neighbors_adaptive: float,
    num_nodes: int,
    max_cutoff: float,
    min_cutoff: float = DEFAULT_MIN_PROBE_CUTOFF,
    cutoff_width: float = DEFAULT_EFFECTIVE_NUM_NEIGHBORS_WIDTH,
    probe_spacing: Optional[float] = None,
    weight_width: Optional[float] = None,
    nl_cutoff: Optional[float] = None,
) -> torch.Tensor:
    """
    Reference (legacy) adaptive-cutoff implementation, preserved for comparison.

    Builds a discrete probe-cutoff grid, evaluates the smoothed neighbor count
    on it, and returns a Gaussian-weighted average of the probes whose neighbor
    counts are closest to ``num_neighbors_adaptive``. Superseded by
    :func:`get_adaptive_cutoffs`, which solves the same selection as a root-find
    by bisection.

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
    :param nl_cutoff: Optional smooth-taper distance applied to each edge's
        contribution to the probe-neighbor counts. Use this when the host
        neighbor list is tighter than ``max_cutoff`` so that an edge whose
        distance crosses ``nl_cutoff`` in MD has its contribution to every
        probe go to zero continuously, removing the discontinuity that would
        otherwise be caused by the edge appearing/disappearing from the NL.
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
            nl_cutoff=nl_cutoff,
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
    nl_cutoff: Optional[float] = None,
) -> torch.Tensor:
    """
    Computes the effective number of neighbors for each probe cutoff.

    :param edge_distances: Distances between centers and their neighbors.
    :param probe_cutoffs: Probe cutoff distances.
    :param centers: Indices of the center atoms.
    :param num_nodes: Total number of center atoms.
    :param width: Width of the cutoff function.
    :param nl_cutoff: Optional smooth-taper distance applied to each edge.
        When provided, each edge's contribution to every probe is multiplied
        by ``cutoff_func(d, nl_cutoff, width)`` so an edge whose distance
        crosses ``nl_cutoff`` (e.g. enters/leaves the host neighbor list in
        MD) has its contribution to every probe go to zero continuously.
    :return: Effective number of neighbors for each center atom and probe cutoff.
    """

    weights = cutoff_func(
        edge_distances.unsqueeze(0), probe_cutoffs.unsqueeze(1), width
    )

    if nl_cutoff is not None:
        # Scalar tensor matching edge_distances dtype/device.
        nl_cutoff_t = edge_distances.new_full((), nl_cutoff)
        nl_taper = cutoff_func(edge_distances, nl_cutoff_t, width)
        weights = weights * nl_taper.unsqueeze(0)

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
