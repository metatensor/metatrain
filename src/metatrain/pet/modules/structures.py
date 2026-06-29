from typing import List, Optional, Tuple

import torch
from metatensor.torch import Labels
from metatomic.torch import NeighborListOptions, System

from .adaptive_cutoff import get_adaptive_cutoffs_grid, get_adaptive_cutoffs_solver
from .nef import (
    compute_reversed_neighbor_list,
    edge_array_to_nef,
    get_corresponding_edges,
    get_nef_indices,
)
from .utilities import cutoff_func_bump, cutoff_func_cosine


def get_pair_sample_labels(
    sample_labels: Labels,
    centers: torch.Tensor,
    neighbors: torch.Tensor,
    cell_shifts: torch.Tensor,
) -> Labels:
    """
    Create per-pair sample labels from center and neighbor atom indices and cell shifts.

    Each row in the returned Labels corresponds to one directed edge (center → neighbor)
    in the neighbor list, identified in the same way as a standard metatensor neighbor
    list: by system index, center atom index, neighbor atom index, and the integer cell
    shift vector ``(cell_shift_a, cell_shift_b, cell_shift_c)``.

    :param sample_labels: Labels for all atoms in the batch, with dimensions
        ``["system", "atom"]``, as returned by :func:`systems_to_batch`.
    :param centers: Flat tensor of center atom global indices, shape ``(n_edges,)``.
        These are post-adaptive-cutoff indices into the concatenated atom list.
    :param neighbors: Flat tensor of neighbor atom global indices, shape ``(n_edges,)``.
        These are post-adaptive-cutoff indices into the concatenated atom list.
    :param cell_shifts: Integer cell shift vectors for each edge, shape ``(n_edges,
        3)``. These are the post-adaptive-cutoff cell shifts matching
        ``centers``/``neighbors``.
    :return: Labels with columns ``["system", "first_atom", "second_atom",
        "cell_shift_a", "cell_shift_b", "cell_shift_c"]``, shape ``(n_edges, 6)``.
    """
    sample_values = sample_labels.values  # (n_atoms, 2): [system, atom]
    center_values = sample_values[centers]  # (n_edges, 2): [system, first_atom]
    neighbor_values = sample_values[neighbors]  # (n_edges, 2): [system, second_atom]

    pair_values = torch.cat(
        [
            center_values[:, :1],  # system        (n_edges, 1)
            center_values[:, 1:],  # first_atom    (n_edges, 1)
            neighbor_values[:, 1:],  # second_atom   (n_edges, 1)
            cell_shifts,  # a, b, c       (n_edges, 3)
        ],
        dim=1,
    )

    return Labels(
        names=[
            "system",
            "first_atom",
            "second_atom",
            "cell_shift_a",
            "cell_shift_b",
            "cell_shift_c",
        ],
        values=pair_values,
    )


def concatenate_structures(
    systems: List[System],
    neighbor_list_options: NeighborListOptions,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Labels,
]:
    """
    Concatenate a list of systems into a single batch.

    :param systems: List of systems to concatenate.
    :param neighbor_list_options: Options for the neighbor list.
    :return: A tuple containing the concatenated positions, centers, neighbors,
        species, cells, cell shifts, system indices, and sample labels.
    """

    device = systems[0].positions.device

    positions_list: List[torch.Tensor] = []
    species_list: List[torch.Tensor] = []
    cells_list: List[torch.Tensor] = []
    nl_values_list: List[torch.Tensor] = []
    sizes: List[int] = []
    num_edges: List[int] = []
    node_offsets_list: List[int] = []

    node_counter = 0
    for system in systems:
        assert len(system.known_neighbor_lists()) >= 1, "no neighbor list found"
        neighbor_list = system.get_neighbor_list(neighbor_list_options)
        nl_values = neighbor_list.samples.values

        positions_list.append(system.positions)
        species_list.append(system.types)
        cells_list.append(system.cell)
        nl_values_list.append(nl_values)

        system_size = len(system)
        node_offsets_list.append(node_counter)
        sizes.append(system_size)
        num_edges.append(nl_values.shape[0])
        node_counter += system_size

    positions = torch.cat(positions_list)
    species = torch.cat(species_list)
    cells = torch.stack(cells_list)
    nl_values = torch.cat(nl_values_list)

    centers = nl_values[:, 0]
    neighbors = nl_values[:, 1]
    cell_shifts = nl_values[:, 2:]

    num_systems = len(systems)
    total_edges = sum(num_edges)
    sizes_tensor = torch.tensor(sizes, device=device, dtype=torch.long)
    num_edges_tensor = torch.tensor(num_edges, device=device, dtype=torch.long)
    node_offsets = torch.tensor(node_offsets_list, device=device, dtype=torch.long)

    edge_offsets = torch.repeat_interleave(
        node_offsets, num_edges_tensor, output_size=total_edges
    ).to(dtype=centers.dtype)
    centers = centers + edge_offsets
    neighbors = neighbors + edge_offsets

    system_indices = torch.repeat_interleave(
        torch.arange(num_systems, device=device),
        sizes_tensor,
        output_size=node_counter,
    )
    atom_indices = torch.arange(
        node_counter, device=device, dtype=torch.long
    ) - torch.repeat_interleave(node_offsets, sizes_tensor, output_size=node_counter)

    sample_values = torch.stack([system_indices, atom_indices], dim=1)
    sample_labels = Labels(
        names=["system", "atom"],
        values=sample_values,
        assume_unique=True,
    )

    return (
        positions,
        centers,
        neighbors,
        species,
        cells,
        cell_shifts,
        system_indices,
        sample_labels,
    )


def compute_batch_tensors(
    positions: torch.Tensor,
    centers: torch.Tensor,
    neighbors: torch.Tensor,
    species: torch.Tensor,
    cells: torch.Tensor,
    cell_shifts: torch.Tensor,
    system_indices: torch.Tensor,
    species_to_species_index: torch.Tensor,
    cutoff: float,
    cutoff_function: str,
    cutoff_width: float,
    num_neighbors_adaptive: Optional[float] = None,
    adaptive_cutoff_method: str = "solver",
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Convert the concatenated, pure-tensor structure representation into the per-edge
    NEF batch tensors required by the PET featurizer.

    This is the pure-PyTorch tail of the structure preprocessing: it takes the plain
    tensors produced by :func:`concatenate_structures` (which is where the metatomic
    ``System`` objects are read) and computes edge vectors, optional adaptive cutoffs,
    cutoff factors, and the NEF (node-edge-feature) reshaping. It contains no
    metatensor / metatomic objects so that it can be ``torch.compile``-d.

    :param positions: Concatenated atomic positions, shape ``(num_nodes, 3)``.
    :param centers: Flat center atom global indices for each edge, shape ``(n_edges,)``.
    :param neighbors: Flat neighbor atom global indices for each edge, shape
        ``(n_edges,)``.
    :param species: Concatenated atomic species, shape ``(num_nodes,)``.
    :param cells: Stacked cell tensors, shape ``(num_systems, 3, 3)``.
    :param cell_shifts: Integer cell shift vectors per edge, shape ``(n_edges, 3)``.
    :param system_indices: System index for each atom, shape ``(num_nodes,)``.
    :param species_to_species_index: Mapping from atomic species to species indices.
    :param cutoff: Neighbor list cutoff radius.
    :param cutoff_function: Type of the smoothing function at the cutoff.
    :param cutoff_width: Width of the cutoff function for a cutoff mask.
    :param num_neighbors_adaptive: Optional maximum number of neighbors per atom.
        If provided, the adaptive cutoff scheme will be used for each atom to
        approximately select this number of neighbors.
    :param adaptive_cutoff_method: Algorithm used to compute the per-atom adaptive
        cutoffs when ``num_neighbors_adaptive`` is set. ``"grid"`` uses the legacy
        probe-grid + Gaussian-weighted average; ``"solver"`` uses a Newton-bisection
        root finder on the smoothed neighbor count.
    :return: A tuple containing the batch tensors:
        - `element_indices_nodes`: The atomic species of the central atoms
        - `element_indices_neighbors`: The atomic species of the neighboring atoms
        - `edge_vectors`: The cartesian edge vectors between the central atoms and their
            neighbors
        - `edge_distances`: The distances between the central atoms and their neighbors
        - `padding_mask`: A padding mask indicating which neighbors are real, and which
            are padded
        - `reverse_neighbor_index`: The reversed neighbor list for each central atom
        - `cutoff_factors`: The cutoff function values for each edge
        - `atomic_cutoffs_stats`: Diagnostic per-atom cutoff radius (detached
            from the autograd graph). With adaptive cutoff active this is
            the per-atom adapted cutoff; otherwise every entry equals
            ``cutoff``. Always shape ``(num_nodes,)``.
        - `centers`: Flat tensor of center atom global indices for each real
          (non-padded) edge, shape ``(n_edges,)``. Suitable for use with
          :func:`get_pair_sample_labels`.
        - `neighbors`: Flat tensor of neighbor atom global indices for each real
          (non-padded) edge, shape ``(n_edges,)``. Suitable for use with
          :func:`get_pair_sample_labels`.
        - `nef_to_edges_neighbor`: Index tensor of shape ``(n_edges,)`` such that
          `nef_tensor[centers, nef_to_edges_neighbor]` recovers the flat edge array
          from a NEF-format tensor. Needed to flatten 3D (edge-like) hook outputs back
          to per-edge arrays for TensorMap construction.
        - `cell_shifts`: Integer cell shift vectors for each real (non-padded) edge,
          shape ``(n_edges, 3)``. Columns correspond to ``(cell_shift_a, cell_shift_b,
          cell_shift_c)``. Suitable for use with :func:`get_pair_sample_labels`.

    """
    # somehow the backward of this operation is very slow at evaluation,
    # where there is only one cell, therefore we simplify the calculation
    # for that case
    if len(cells) == 1:
        cell_contributions = cell_shifts.to(cells.dtype) @ cells[0]
    else:
        cell_contributions = torch.einsum(
            "ab, abc -> ac",
            cell_shifts.to(cells.dtype),
            cells[system_indices[centers]],
        )
    edge_vectors = positions[neighbors] - positions[centers] + cell_contributions
    edge_distances = torch.norm(edge_vectors, dim=-1) + 1e-15

    num_nodes = len(positions)

    if num_neighbors_adaptive is not None:
        with torch.profiler.record_function("PET::get_adaptive_cutoffs"):
            # Adaptive cutoff scheme to approximately select `num_neighbors_adaptive`
            # neighbors for each atom
            if adaptive_cutoff_method.lower() == "solver":
                atomic_cutoffs = get_adaptive_cutoffs_solver(
                    centers,
                    edge_distances,
                    num_neighbors_adaptive,
                    num_nodes,
                    cutoff,
                    cutoff_width=cutoff_width,
                )
            elif adaptive_cutoff_method.lower() == "grid":
                atomic_cutoffs = get_adaptive_cutoffs_grid(
                    centers,
                    edge_distances,
                    num_neighbors_adaptive,
                    num_nodes,
                    cutoff,
                    cutoff_width=cutoff_width,
                )
            else:
                raise ValueError(
                    "adaptive_cutoff_method must be 'grid' or 'solver', got "
                    + adaptive_cutoff_method
                )
            atomic_cutoffs_stats = atomic_cutoffs.detach()
            # Symmetrize the cutoffs between pairs of atoms (PET needs this symmetry
            # due to its corresponding edge indexing ij -> ji)
            pair_cutoffs = (atomic_cutoffs[centers] + atomic_cutoffs[neighbors]) / 2.0
        with torch.profiler.record_function("PET::adaptive_cutoff_masking"):
            keep = torch.nonzero(edge_distances <= pair_cutoffs).squeeze(-1)
            pair_cutoffs = pair_cutoffs.index_select(0, keep)
            centers = centers.index_select(0, keep)
            neighbors = neighbors.index_select(0, keep)
            edge_vectors = edge_vectors.index_select(0, keep)
            cell_shifts = cell_shifts.index_select(0, keep)
            edge_distances = edge_distances.index_select(0, keep)
    else:
        pair_cutoffs = cutoff * torch.ones(
            len(centers), device=positions.device, dtype=positions.dtype
        )
        atomic_cutoffs_stats = cutoff * torch.ones(
            num_nodes, device=positions.device, dtype=positions.dtype
        )

    # ``torch.bincount`` has a data-dependent output shape, which becomes an
    # unbacked symbolic dimension under ``torch.compile`` and corrupts the whole NEF
    # grid. ``scatter_add`` into a ``num_nodes``-sized buffer is equivalent but keeps a
    # static (backed) shape.
    num_neighbors = torch.zeros(
        num_nodes, dtype=centers.dtype, device=centers.device
    ).scatter_add_(0, centers, torch.ones_like(centers))
    # ``max_edges_per_node`` (the largest neighbour count of any atom) becomes the size
    # of the NEF grid's second dimension. The ``numel`` guard keeps empty systems
    # (no atoms) well defined.

    max_edges_per_node = (
        int(torch.max(num_neighbors)) if num_neighbors.numel() > 0 else 0
    )

    # Tell the compiler this data-dependent scalar is a valid (non-negative) size, so
    # the symbolic NEF dimension is well defined. ``torch._check`` is a compile-only
    # hint and not a TorchScript builtin, so it must be dead-code-eliminated under
    # scripting. ``torch.jit.is_scripting()`` is the guard TorchScript recognises and
    # prunes (``torch.compiler.is_compiling()`` is not, so it would try to script the
    # unknown op). ``_check(x >= 0)`` is the forward-compatible replacement for the
    # now-deprecated ``torch._check_is_size``.
    if not torch.jit.is_scripting():
        torch._check(max_edges_per_node >= 0)

    if cutoff_function.lower() == "bump":
        # use bump switching function for adaptive cutoff
        cutoff_factors = cutoff_func_bump(edge_distances, pair_cutoffs, cutoff_width)
    elif cutoff_function.lower() == "cosine":
        # backward-compatible cosine swithcing for fixed cutoff
        cutoff_factors = cutoff_func_cosine(edge_distances, pair_cutoffs, cutoff_width)
    else:
        raise ValueError(
            f"Unknown cutoff function type: {cutoff_function}. "
            f"Supported types are 'Cosine' and 'Bump'."
        )

    # Convert to NEF (Node-Edge-Feature) format:
    # Pass `num_neighbors` in so `get_nef_indices` doesn't re-run bincount.
    nef_indices, nef_to_edges_neighbor, nef_mask = get_nef_indices(
        centers, num_neighbors, max_edges_per_node
    )

    # Element indices
    element_indices_nodes = species_to_species_index[species]
    element_indices_neighbors = element_indices_nodes[neighbors]

    # Send everything to NEF:
    edge_vectors = edge_array_to_nef(edge_vectors, nef_indices)
    edge_distances = torch.sqrt(torch.sum(edge_vectors**2, dim=2) + 1e-15)
    element_indices_neighbors = edge_array_to_nef(
        element_indices_neighbors, nef_indices
    )
    cutoff_factors = edge_array_to_nef(cutoff_factors, nef_indices, nef_mask, 0.0)

    corresponding_edges = get_corresponding_edges(centers, neighbors, cell_shifts)

    # These are the two arrays we need for message passing with edge reversals,
    # if indexing happens in a two-dimensional way:
    # edges_ji = edges_ij[reversed_neighbor_list, neighbors_index]
    reversed_neighbor_list = compute_reversed_neighbor_list(
        nef_indices, corresponding_edges, nef_to_edges_neighbor, nef_mask
    )
    neighbors_index = edge_array_to_nef(neighbors, nef_indices).to(torch.int64)

    # Here, we compute the array that allows indexing into a flattened
    # version of the edge array (where the first two dimensions are merged):
    reverse_neighbor_index = (
        neighbors_index * neighbors_index.shape[1] + reversed_neighbor_list
    )
    # At this point, we have `reverse_neighbor_index[~nef_mask] = 0`, which however
    # creates too many of the same index which slows down backward enormously.
    # (See see https://github.com/pytorch/pytorch/issues/41162)
    # We therefore replace the padded indices with a sequence of unique indices.
    # ``cumsum(~mask) - 1`` gives, at each padded slot, its running position among the
    # padded slots in flattened (row-major) order — identical to assigning
    # ``arange(num_padded)`` via a boolean mask, but with static shapes. This avoids a
    # data-dependent kernel and lets the preprocessing be captured by ``torch.compile``.
    flat_mask = nef_mask.reshape(-1)
    flat_reverse = reverse_neighbor_index.reshape(-1)
    padded_unique = torch.cumsum((~flat_mask).to(torch.long), dim=0) - 1
    flat_reverse = torch.where(flat_mask, flat_reverse, padded_unique)
    reverse_neighbor_index = flat_reverse.reshape(reverse_neighbor_index.shape)

    return (
        element_indices_nodes,
        element_indices_neighbors,
        edge_vectors,
        edge_distances,
        nef_mask,
        reverse_neighbor_index,
        cutoff_factors,
        atomic_cutoffs_stats,
        centers,
        neighbors,
        nef_to_edges_neighbor,
        cell_shifts,
    )
