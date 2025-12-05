from typing import Optional

import torch

from .utilities import cutoff_func


DEFAULT_MIN_PROBE_CUTOFF = 0.5
DEFAULT_EFFECTIVE_NUM_NEIGHBORS_WIDTH = 0.5
DEFAULT_PROBE_CUTOFFS_SPACING = 0.1


def get_adaptive_cutoffs(
    centers: torch.Tensor,
    edge_distances: torch.Tensor,
    max_num_neighbors: float,
    num_nodes: int,
    max_cutoff: float,
    grid_spacing: float = DEFAULT_PROBE_CUTOFFS_SPACING,
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
        DEFAULT_MIN_PROBE_CUTOFF,
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
        cutoffs_weights = get_gaussian_cutoff_weights(
            effective_num_neighbors,
            max_num_neighbors,
            num_nodes,
            probe_cutoffs=probe_cutoffs,
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
    :param width: Width of the cutoff function. If None, it will be
        automatically determined from the probe cutoff spacing.
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
    # Vectorized version: use scatter_add_ to accumulate weights for all probe
    # cutoffs at once
    centers_expanded = (
        centers.unsqueeze(0).expand(len(probe_cutoffs), -1).to(torch.int64)
    )
    probe_num_neighbors.scatter_add_(1, centers_expanded, weights)
    probe_num_neighbors = probe_num_neighbors.T.contiguous()
    return probe_num_neighbors  # / 0.286241 # normalization factor to account for the form of the cutoff function


def get_gaussian_cutoff_weights(
    effective_num_neighbors: torch.Tensor,
    max_num_neighbors: float,
    width: Optional[float] = None,
    probe_cutoffs: Optional[torch.Tensor] = None,
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
    if width is None:
        assert probe_cutoffs is not None, (
            "Either width or probe_cutoffs must be provided."
        )
        # Automatically determine width from probe cutoff spacing
        delta_r = probe_cutoffs[1] - probe_cutoffs[0]
        width = 3 * max_num_neighbors * delta_r.item() / max(probe_cutoffs).item()

    max_num_neighbors_t = torch.as_tensor(
        max_num_neighbors, device=effective_num_neighbors.device
    )

    diff = effective_num_neighbors - max_num_neighbors_t
    x = torch.linspace(
        0, 1, effective_num_neighbors.shape[1], device=effective_num_neighbors.device
    )
    baseline = max_num_neighbors_t * x**3

    diff = diff + baseline.unsqueeze(0)

    weights = torch.exp(-0.5 * (diff / width) ** 2)

    # row-wise normalization, with small epsilon to avoid division by zero
    weights_sum = weights.sum(dim=1, keepdim=True)
    weights = weights / weights_sum

    return weights
