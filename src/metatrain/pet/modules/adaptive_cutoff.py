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
) -> torch.Tensor:
    """
    Computes the adaptive cutoff values for each center atom.

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

    # heuristic for the Gaussian weight width. this is chosen to ensure that
    # for typical neighbor distributions the weights are non-zero for multiple
    # probe cutoffs
    # if weight_width is None:
    #    weight_width = 3 * num_neighbors_adaptive * probe_spacing / max_cutoff
    #
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
            width_t = diff * 0.5
        elif diff.shape[-1] == 2:
            w = (diff[..., 1] - diff[..., 0]).abs().clamp_min(eps)
            width_t = torch.stack([w, w], dim=-1)
        else:
            width_t = torch.empty_like(diff)
            # centered difference
            width_t[..., 1:-1] = 0.5 * (diff[..., 2:] - diff[..., :-2])
            width_t[..., 0] = diff[..., 1] - diff[..., 0]  # forward diff
            width_t[..., -1] = diff[..., -1] - diff[..., -2]  # backward diff
            width_t = width_t.abs().clamp_min(eps)
    else:
        width_t = torch.ones_like(diff) * width

    logw = -0.5 * (diff / width_t) ** 2

    weights = torch.exp(logw - logw.max())

    # row-wise normalization of the weights
    weights_sum = weights.sum(dim=1, keepdim=True)
    weights = weights / weights_sum

    return weights
