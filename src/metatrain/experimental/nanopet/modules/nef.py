from typing import Optional

import torch


def edge_array_to_nef(
    edge_array,
    nef_indices,
    mask: Optional[torch.Tensor] = None,
    fill_value: float = 0.0,
):
    """Converts an edge array to a NEF array."""

    if mask is None:
        return edge_array[nef_indices]
    else:
        return torch.where(
            mask.reshape(mask.shape + (1,) * (len(edge_array.shape) - 1)),
            edge_array[nef_indices],
            fill_value,
        )


def nef_array_to_edges(nef_array, centers, nef_to_edges_neighbor):
    """Converts a NEF array to an edge array."""

    return nef_array[centers, nef_to_edges_neighbor]
