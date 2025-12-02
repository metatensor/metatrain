from typing import Optional

import torch

from .utilities import smooth_delta_function, step_characteristic_function


DEFAULT_MIN_PROBE_CUTOFF = 0.5
DEFAULT_PROBE_CUTOFFS_SPACING = 0.1


def get_adaptive_cutoffs(
    centers: torch.Tensor,
    edge_distances: torch.Tensor,
    max_num_neighbors: float,
    num_nodes: int,
    max_cutoff: float,
    grid_spacing: float = DEFAULT_PROBE_CUTOFFS_SPACING,
    weighting: str = "gaussian",
) -> torch.Tensor:
    """
    Computes the adaptive cutoff values for each center atom.

    :param centers: Indices of the center atoms.
    :param edge_distances: Distances between centers and their neighbors.
    :param max_num_neighbors: Target maximum number of neighbors per atom.
    :param num_nodes: Total number of center atoms.
    :param max_cutoff: Maximum cutoff distance to consider.
    :param grid_spacing: Spacing between probe cutoff distances.
    :param weighting: Weighting scheme to use ('gaussian' or 'exponential').
    :return: Adapted cutoff distances for each center atom.
    """
    probe_cutoffs = torch.arange(
        0.5,
        max_cutoff,
        grid_spacing,
        device=edge_distances.device,
        dtype=edge_distances.dtype,
    )
    with torch.profiler.record_function("PET::get_effective_num_neighbors"):
        effective_num_neighbors = get_effective_num_neighbors(
            edge_distances,
            probe_cutoffs,
            centers,
            num_nodes,
        )
    with torch.profiler.record_function("PET::get_cutoff_weights"):
        if weighting == "gaussian":
            cutoffs_weights = get_gaussian_cutoff_weights(
                effective_num_neighbors, probe_cutoffs, max_num_neighbors, num_nodes
            )
        elif weighting == "exponential":
            cutoffs_weights = get_exponential_cutoff_weights(
                effective_num_neighbors, probe_cutoffs, max_num_neighbors
            )
        else:
            raise ValueError(
                f"Unknown weighting scheme: {weighting}"
                " Supported: 'gaussian', 'exponential'."
            )
    with torch.profiler.record_function("PET::calculate_adapted_cutoffs"):
        adapted_atomic_cutoffs = probe_cutoffs @ cutoffs_weights.T
    return adapted_atomic_cutoffs


def get_effective_num_neighbors(
    edge_distances: torch.Tensor,
    probe_cutoffs: torch.Tensor,
    centers: torch.Tensor,
    num_nodes: int,
    width: Optional[float] = None,
) -> torch.Tensor:
    """
    Computes the effective number of neighbors for each probe cutoff.

    :param edge_distances: Distances between centers and their neighbors.
    :param probe_cutoffs: Probe cutoff distances.
    :param centers: Indices of the center atoms.
    :param num_nodes: Total number of center atoms.
    :param width: Width of the cutoff function. If None, it will be
        automatically determined from the probe cutoff spacing.
    :return: Effective number of neighbors for each center atom and probe cutoff.
    """
    if width is None:
        # Automatically determine width from probe cutoff spacing
        # Use 2.5x the spacing for a smooth step function
        if len(probe_cutoffs) > 1:
            probe_spacing = probe_cutoffs[1] - probe_cutoffs[0]
            width = 2.5 * probe_spacing
        else:
            width = 0.5  # fallback for single probe cutoff

    weights = step_characteristic_function(
        edge_distances.unsqueeze(0), probe_cutoffs.unsqueeze(1), width
    )
    probe_num_neighbors = torch.zeros(
        (len(probe_cutoffs), num_nodes),
        dtype=edge_distances.dtype,
        device=edge_distances.device,
    )
    # Vectorized version: use scatter_add_ to accumulate weights for all probe cutoffs at once
    centers_expanded = centers.unsqueeze(0).expand(len(probe_cutoffs), -1)
    probe_num_neighbors.scatter_add_(1, centers_expanded, weights)
    probe_num_neighbors = probe_num_neighbors.T.contiguous()
    return probe_num_neighbors


def get_gaussian_cutoff_weights(
    effective_num_neighbors: torch.Tensor,
    probe_cutoffs: torch.Tensor,
    max_num_neighbors: float,
    num_nodes: int,
    width: float = 0.5,
) -> torch.Tensor:
    """
    Computes the weights for each probe cutoff based on
    the effective number of neighbors using Gaussian weights
    centered at the expected number of neighbors.

    :param effective_num_neighbors: Effective number of neighbors for each center atom
        and probe cutoff.
    :param probe_cutoffs: Probe cutoff distances.
    :param max_num_neighbors: Target maximum number of neighbors per atom.
    :param num_nodes: Total number of center atoms.
    :param width: Width of the Gaussian function.
    :return: Weights for each probe cutoff.
    """
    num_neighbors_threshold = (
        max_num_neighbors
        * torch.ones(num_nodes, 1, device=effective_num_neighbors.device)
        - 5e-2
    )  # eps
    cutoffs_threshold_idx = torch.searchsorted(
        effective_num_neighbors, num_neighbors_threshold, right=False
    ).clamp(max=len(probe_cutoffs) - 1)
    cutoffs_thresholds = probe_cutoffs[cutoffs_threshold_idx]

    cutoffs_weights = smooth_delta_function(
        probe_cutoffs.unsqueeze(0), cutoffs_thresholds, width=width
    )
    cutoffs_weights = cutoffs_weights / cutoffs_weights.sum(dim=1, keepdim=True)
    return cutoffs_weights


def get_exponential_cutoff_weights(
    effective_num_neighbors: torch.Tensor,
    probe_cutoffs: torch.Tensor,
    max_num_neighbors: float,
    width: float = 0.5,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    Computes the weights for each probe cutoff based on
    the effective number of neighbors using Exponential weights.

    :param effective_num_neighbors: Effective number of neighbors for each center atom
        and probe cutoff.
    :param probe_cutoffs: Probe cutoff distances.
    :param max_num_neighbors: Target maximum number of neighbors per atom.
    :param width: Width of the step characteristic function.
    :param beta: Exponential scaling factor.
    :return: Weights for each probe cutoff.
    """
    max_num_neighbors = torch.tensor(
        max_num_neighbors,
        device=effective_num_neighbors.device,
        dtype=effective_num_neighbors.dtype,
    )
    cutoffs_weights = torch.exp(beta * probe_cutoffs) * step_characteristic_function(
        effective_num_neighbors, max_num_neighbors, width=width
    )
    cutoffs_weights = cutoffs_weights / cutoffs_weights.sum(dim=1, keepdim=True)
    return cutoffs_weights
