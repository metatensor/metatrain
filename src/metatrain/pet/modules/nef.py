"""
Module with functions to manipulate NEF (Node Edge Feature) arrays.

The NEF representation is what the internals of PET use.
In the NEF representation, the first dimension is the center node
(i.e. the "i" node in an "i -> j" edge), and the second dimension
is the edges for that node. Not all center nodes have the same number
of edges, so padding is used to ensure that all nodes have the same
number of edges.

Most of the functions have the purpose of converting between
edge arrays with shape (n_edges, ...) and NEF arrays with shape
(n_nodes, n_edges_per_node, ...).
"""

from typing import List, Optional, Tuple

import torch


def get_nef_indices(
    centers: torch.Tensor, n_nodes: int, n_edges_per_node: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes tensors of indices useful to convert between edge
    and NEF layouts; the usage and function of `nef_indices` and
    `nef_to_edges_neighbor` is clear in the ``edge_array_to_nef``
    and ``nef_array_to_edges`` functions below.

    :param centers: A 1D tensor of shape (n_edges,) containing the
        indices of the center nodes for each edge, with the center nodes
        being the "i" node in an "i -> j" edge.
    :param n_nodes: The number of nodes in the graph.
    :param n_edges_per_node: The maximum number of edges per node.

    :return: A tuple with three tensors (nef_indices, nef_to_edges_neighbor, nef_mask).
        In particular:
        nef_array = edge_array[nef_indices]
        edge_array = nef_array[centers, nef_to_edges_neighbor]
        The third output, nef_mask, is a mask that can be used to
        filter out the padding values in the NEF array, as different
        nodes will have, in general, different number of edges.
    """

    bincount = torch.bincount(centers, minlength=n_nodes)

    arange = torch.arange(n_edges_per_node, device=centers.device)
    arange_expanded = arange.view(1, -1).expand(n_nodes, -1)
    nef_mask = arange_expanded < bincount.view(-1, 1)

    argsort = torch.argsort(centers, stable=True)

    nef_indices = torch.zeros(
        (n_nodes, n_edges_per_node), dtype=torch.long, device=centers.device
    )
    nef_indices[nef_mask] = argsort

    nef_to_edges_neighbor = torch.empty_like(centers, dtype=torch.long)
    nef_to_edges_neighbor[argsort] = arange_expanded[nef_mask]

    return nef_indices, nef_to_edges_neighbor, nef_mask


def get_corresponding_edges(array: torch.Tensor) -> torch.Tensor:
    """
    Computes the corresponding edge (i.e., the edge that goes in the
    opposite direction) for each edge in the array; this is useful
    in the message-passing operation.

    :param array: A 2D tensor of shape (n_edges, 5). For each i -> j
        edge, the first column contains the index of the center node i,
        the second column contains the index of the neighbor node j,
        and the last three columns contain the cell shifts along x, y, and z
        directions, respectively.

    :return: A 1D tensor of shape (n_edges,) containing, for each edge,
        the index of the corresponding edge (i.e., the edge that goes
        in the opposite direction). If the input array is empty, an
        empty tensor is returned.
    """

    if array.numel() == 0:
        return torch.empty((0,), dtype=array.dtype, device=array.device)

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
    edge_array: torch.Tensor,
    nef_indices: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Converts an edge array to a NEF array.

    :param edge_array: A tensor where the first dimension is the index of
        the edge, i.e. with shape (n_edges, ...).
    :param nef_indices: The indices to convert from edge to NEF layout,
        as returned by the ``get_nef_indices`` function.
    :param mask: An optional boolean mask of shape (n_nodes, n_edges_per_node),
        as returned by the ``get_nef_indices`` function. If provided,
        the output NEF array will have the values in the positions
        where the mask is False set to ``fill_value``.
    :param fill_value: The value to use to fill the positions in the
        NEF array where the mask is False. Only used if ``mask`` is
        provided.

    :return: A tensor with the same information as ``edge_array``,
        but in NEF layout, i.e. with shape (n_nodes, n_edges_per_node, ...).
        If ``mask`` is provided, the values in the positions where
        the mask is False are set to ``fill_value``.
    """
    if mask is None:
        return edge_array[nef_indices]
    else:
        return torch.where(
            mask.reshape(mask.shape + (1,) * (len(edge_array.shape) - 1)),
            edge_array[nef_indices],
            fill_value,
        )


def nef_array_to_edges(
    nef_array: torch.Tensor, centers: torch.Tensor, nef_to_edges_neighbor: torch.Tensor
) -> torch.Tensor:
    """Converts a NEF array to an edge array.

    :param nef_array: A tensor where the first two dimensions are the
        indices of the NEF layout, i.e. with shape (n_nodes, n_edges_per_node, ...).
    :param centers: The indices of the center nodes for each edge.
    :param nef_to_edges_neighbor: The indices of the edges for each
        neighbor in the NEF layout, as returned by the ``get_nef_indices`` function.

    :return: A tensor with the same information as ``nef_array``,
        but in edge layout, i.e. with shape (n_edges, ...).
    """
    return nef_array[centers, nef_to_edges_neighbor]


def compute_reversed_neighbor_list(
    nef_indices: torch.Tensor,
    corresponding_edges: torch.Tensor,
    nef_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Creates a reversed neighborlist, where for each
    center atom `i` and its neighbor `j` in the original
    neighborlist, the position of atom `i` in the list
    of neighbors of atom `j` is returned.

    :param nef_indices: The indices to convert from edge to NEF layout,
        as returned by the ``get_nef_indices`` function.
    :param corresponding_edges: The indices of the corresponding edges,
        as returned by the ``get_corresponding_edges`` function.
    :param nef_mask: A boolean mask of shape (n_nodes, n_edges_per_node),
        as returned by the ``get_nef_indices`` function.
    :return: A tensor of the same shape as ``nef_indices``,
        where each entry contains the position of the center
        atom in the neighborlist of the corresponding neighbor atom.
    """
    num_atoms, max_num_neighbors = nef_indices.shape

    flat_edge_indices = nef_indices.reshape(-1)
    flat_positions = torch.arange(max_num_neighbors, device=nef_indices.device).repeat(
        num_atoms
    )
    flat_mask = nef_mask.reshape(-1)

    if flat_edge_indices.numel() == 0:
        max_edge_index = 0
    else:
        max_edge_index = int(flat_edge_indices.max().item()) + 1
    size: List[int] = [max_edge_index]

    edge_index_to_position = torch.full(
        size,
        0,
        dtype=torch.long,
        device=nef_indices.device,
    )
    edge_index_to_position[flat_edge_indices[flat_mask]] = flat_positions[flat_mask]

    reverse_edge_idx = corresponding_edges[nef_indices]
    reversed_neighbor_list = edge_index_to_position[reverse_edge_idx]
    reversed_neighbor_list = reversed_neighbor_list.masked_fill(~nef_mask, 0)

    return reversed_neighbor_list
