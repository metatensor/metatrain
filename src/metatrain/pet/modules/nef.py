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

from typing import Optional, Tuple

import torch


def get_nef_indices(
    centers: torch.Tensor, num_neighbors: torch.Tensor, n_edges_per_node: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes tensors of indices useful to convert between edge
    and NEF layouts; the usage and function of `nef_indices` and
    `nef_to_edges_neighbor` is clear in the ``edge_array_to_nef``
    and ``nef_array_to_edges`` functions below.

    :param centers: A 1D tensor of shape (n_edges,) containing the
        indices of the center nodes for each edge, with the center nodes
        being the "i" node in an "i -> j" edge.
    :param num_neighbors: A 1D tensor of shape (n_nodes,) containing
        the number of neighbors for each node. Typically obtained from
        ``torch.bincount(centers, minlength=n_nodes)`` at the caller,
        where it is already needed to size the NEF grid.
    :param n_edges_per_node: The maximum number of edges per node.

    :return: A tuple with three tensors (nef_indices, nef_to_edges_neighbor, nef_mask).
        In particular:
        nef_array = edge_array[nef_indices]
        edge_array = nef_array[centers, nef_to_edges_neighbor]
        The third output, nef_mask, is a mask that can be used to
        filter out the padding values in the NEF array, as different
        nodes will have, in general, different number of edges.
    """

    n_nodes = num_neighbors.shape[0]

    arange = torch.arange(n_edges_per_node, device=centers.device)
    nef_mask = arange.view(1, -1).expand(n_nodes, -1) < num_neighbors.view(-1, 1)

    argsort = torch.argsort(centers, stable=True)

    n_edges = centers.shape[0]
    sorted_centers = centers.index_select(0, argsort)
    starts = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    position_within = torch.arange(
        n_edges, device=centers.device
    ) - starts.index_select(0, sorted_centers)

    nef_indices = torch.zeros(
        n_nodes * n_edges_per_node, dtype=torch.long, device=centers.device
    )
    flat_target = sorted_centers * n_edges_per_node + position_within
    nef_indices[flat_target] = argsort
    nef_indices = nef_indices.view(n_nodes, n_edges_per_node)

    nef_to_edges_neighbor = torch.empty_like(centers, dtype=torch.long)
    nef_to_edges_neighbor[argsort] = position_within

    return nef_indices, nef_to_edges_neighbor, nef_mask


def get_corresponding_edges(
    centers: torch.Tensor,
    neighbors: torch.Tensor,
    cell_shifts: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the corresponding edge (i.e., the edge that goes in the
    opposite direction) for each edge in the array; this is useful
    in the message-passing operation.

    :param centers: A 1D tensor of shape (n_edges,) containing, for each
        ``i -> j`` edge, the index of the center node ``i``.
    :param neighbors: A 1D tensor of shape (n_edges,) containing, for each
        ``i -> j`` edge, the index of the neighbor node ``j``.
    :param cell_shifts: A 2D tensor of shape (n_edges, 3) with the cell shifts
        along x, y, z for each edge.

    :return: A 1D tensor of shape (n_edges,) containing, for each edge,
        the index of the corresponding edge (i.e., the edge that goes
        in the opposite direction). If the input is empty, an empty
        tensor is returned.
    """

    if centers.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=centers.device)

    centers = centers.to(torch.int64)
    neighbors = neighbors.to(torch.int64)
    cell_shifts = cell_shifts.to(torch.int64)

    min_per_axis = cell_shifts.amin(dim=0)
    cs_norm = cell_shifts - min_per_axis
    neg_cs_norm = -cell_shifts - min_per_axis

    max_per_axis = cs_norm.amax(dim=0) + 1
    max_centers_neighbors = centers.amax() + 1

    size_z = max_per_axis[2]
    size_yz = max_per_axis[1] * size_z
    size_xyz = max_per_axis[0] * size_yz
    size_total = max_centers_neighbors * size_xyz

    unique_id = (
        centers * size_total
        + neighbors * size_xyz
        + cs_norm[:, 0] * size_yz
        + cs_norm[:, 1] * size_z
        + cs_norm[:, 2]
    )
    unique_id_inverse = (
        neighbors * size_total
        + centers * size_xyz
        + neg_cs_norm[:, 0] * size_yz
        + neg_cs_norm[:, 1] * size_z
        + neg_cs_norm[:, 2]
    )

    unique_id_argsort = unique_id.argsort()
    unique_id_inverse_argsort = unique_id_inverse.argsort()

    corresponding_edges = torch.empty_like(centers)
    corresponding_edges[unique_id_argsort] = unique_id_inverse_argsort

    return corresponding_edges


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
    nef_to_edges_neighbor: torch.Tensor,
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
    :param nef_to_edges_neighbor: Per-edge within-row position in the NEF
        grid, as returned by the ``get_nef_indices`` function. Acts directly
        as the ``edge_index -> position`` lookup that the previous
        implementation built on the fly.
    :param nef_mask: A boolean mask of shape (n_nodes, n_edges_per_node),
        as returned by the ``get_nef_indices`` function.
    :return: A tensor of the same shape as ``nef_indices``,
        where each entry contains the position of the center
        atom in the neighborlist of the corresponding neighbor atom.
    """
    reverse_edge_idx = corresponding_edges[nef_indices]
    reversed_neighbor_list = nef_to_edges_neighbor[reverse_edge_idx]
    reversed_neighbor_list = reversed_neighbor_list.masked_fill(~nef_mask, 0)

    return reversed_neighbor_list
