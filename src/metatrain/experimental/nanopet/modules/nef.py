from typing import Optional

import torch


@torch.jit.script
def get_nef_indices(centers, n_nodes: int, n_edges_per_node: int):
    """Transform the center indices into NEF indices."""

    n_edges = len(centers)
    edges_to_nef = torch.zeros(
        (n_nodes, n_edges_per_node), dtype=torch.long, device=centers.device
    )
    nef_to_edges_neighbor = torch.empty(
        (n_edges,), dtype=torch.long, device=centers.device
    )
    node_counter = torch.zeros((n_nodes,), dtype=torch.long, device=centers.device)
    nef_mask = torch.full(
        (n_nodes, n_edges_per_node), 0, dtype=torch.bool, device=centers.device
    )

    for i in range(n_edges):
        center = centers[i]
        edges_to_nef[center, node_counter[center]] = i
        nef_mask[center, node_counter[center]] = True
        nef_to_edges_neighbor[i] = node_counter[center]
        node_counter[center] += 1

    return (edges_to_nef, nef_to_edges_neighbor, nef_mask)


@torch.jit.script
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


@torch.jit.script
def nef_array_to_edges(nef_array, centers, nef_to_edges_neighbor):
    """Converts a NEF array to an edge array."""

    return nef_array[centers, nef_to_edges_neighbor]
