from typing import Optional

import torch


def get_nef_indices(centers, n_nodes: int, n_edges_per_node: int):

    bincount = torch.bincount(centers, minlength=n_nodes)

    arange = torch.arange(n_edges_per_node, device=centers.device)
    arange_expanded = arange.view(1, -1).expand(n_nodes, -1)
    nef_mask = arange_expanded < bincount.view(-1, 1)

    argsort = torch.argsort(centers, stable=True)

    edges_to_nef = torch.zeros(
        (n_nodes, n_edges_per_node), dtype=torch.long, device=centers.device
    )
    edges_to_nef[nef_mask] = argsort

    nef_to_edges_neighbor = torch.empty_like(centers, dtype=torch.long)
    nef_to_edges_neighbor[argsort] = arange_expanded[nef_mask]

    return edges_to_nef, nef_to_edges_neighbor, nef_mask


def get_corresponding_edges(array):
    array = array.to(torch.int64)  # avoid overflow

    centers = array[:, 0]
    neighbors = array[:, 1]
    cell_shifts_x = array[:, 2]
    cell_shifts_y = array[:, 3]
    cell_shifts_z = array[:, 4]

    # will be useful later
    negative_cell_shifts_x = -cell_shifts_x
    negative_cell_shifts_y = -cell_shifts_y
    negative_cell_shifts_z = -cell_shifts_z

    # create a unique identifier for each edge
    # first, we shift the cell_shifts so that the minimum value is 0
    min_cell_shift_x = cell_shifts_x.min()
    cell_shifts_x = cell_shifts_x - min_cell_shift_x
    negative_cell_shifts_x = negative_cell_shifts_x - min_cell_shift_x

    min_cell_shift_y = cell_shifts_y.min()
    cell_shifts_y = cell_shifts_y - min_cell_shift_y
    negative_cell_shifts_y = negative_cell_shifts_y - min_cell_shift_y

    min_cell_shift_z = cell_shifts_z.min()
    cell_shifts_z = cell_shifts_z - min_cell_shift_z
    negative_cell_shifts_z = negative_cell_shifts_z - min_cell_shift_z

    max_centers_neigbors = centers.max() + 1  # same as neighbors.max() + 1
    max_shift_x = cell_shifts_x.max() + 1
    max_shift_y = cell_shifts_y.max() + 1
    max_shift_z = cell_shifts_z.max() + 1

    size_1 = max_shift_z
    size_2 = max_shift_y * size_1
    size_3 = max_shift_x * size_2
    size_4 = max_centers_neigbors * size_3

    unique_id = (
        centers * size_4
        + neighbors * size_3
        + cell_shifts_x * size_2
        + cell_shifts_y * size_1
        + cell_shifts_z
    )

    # the inverse is the same, but centers and neighbors are swapped
    # and we use the negative values of the cell_shifts
    unique_id_inverse = (
        neighbors * size_4
        + centers * size_3
        + negative_cell_shifts_x * size_2
        + negative_cell_shifts_y * size_1
        + negative_cell_shifts_z
    )

    unique_id_argsort = unique_id.argsort()
    unique_id_inverse_argsort = unique_id_inverse.argsort()

    corresponding_edges = torch.empty_like(centers)
    corresponding_edges[unique_id_argsort] = unique_id_inverse_argsort

    return corresponding_edges.to(array.dtype)


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
