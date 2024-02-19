import torch


def get_nef_indices(centers, n_nodes: int, n_edges_per_node: int):
    """Transform the center indices into NEF indices."""

    nef_indices = torch.full((n_nodes, n_edges_per_node), -1, dtype=torch.int64)
    for i in range(n_nodes):
        where = (centers == i).nonzero(as_tuple=False)
        nef_indices[i, : len(where)] = where.squeeze(-1)

    mask = torch.where(nef_indices != -1, 1, 0).bool()
    nef_indices[nef_indices == -1] = 0
    return (nef_indices, mask)


def edge_array_to_nef(edge_array, nef_indices, fill_value):
    """Converts an edge array to a NEF array."""

    mask, nef_indices = nef_indices

    return torch.where(
        mask.reshape(mask.shape + (1,) * (len(edge_array.shape) - 1)),
        edge_array[nef_indices],
        fill_value,
    )
