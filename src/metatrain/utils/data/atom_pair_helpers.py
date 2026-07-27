from typing import Dict, List, Optional, Tuple

import torch
from metatensor.torch import Labels
from metatomic.torch import NeighborListOptions, System

from metatrain.utils.data.target_info import TargetInfo


def check_no_atom_pair_targets(
    targets: Dict[str, TargetInfo], architecture_name: str
) -> None:
    """
    Raise a clear error if any of ``targets`` has ``sample_kind == "atom_pair"``.

    This is used by architectures that do not yet support atom-pair targets.

    :param targets: Dict mapping target names to their :py:class:`TargetInfo`, e.g.
        ``dataset_info.targets``.
    :param architecture_name: Name of the calling architecture, used in the error
        message.
    :raises ValueError: If any target has ``sample_kind == "atom_pair"``.
    """
    unsupported = [
        name for name, info in targets.items() if info.sample_kind == "atom_pair"
    ]
    if unsupported:
        raise ValueError(
            f"the '{architecture_name}' architecture does not yet support "
            f"'atom_pair' sample-kind targets: {unsupported}."
        )


def get_pair_sample_labels(
    sample_labels: Labels,
    centers: Optional[torch.Tensor] = None,
    neighbors: Optional[torch.Tensor] = None,
    cell_shifts: Optional[torch.Tensor] = None,
    systems: Optional[List[System]] = None,
    nl_options: Optional[NeighborListOptions] = None,
) -> Labels:
    """
    Create per-pair sample labels from center and neighbor atom indices and cell shifts.

    Each row in the returned Labels corresponds to one directed edge (center → neighbor)
    in the neighbor list, identified in the same way as a standard metatensor neighbor
    list: by system index, center atom index, neighbor atom index, and the integer cell
    shift vector ``(cell_shift_a, cell_shift_b, cell_shift_c)``.

    Either ``centers``, ``neighbors`` and ``cell_shifts`` must all be provided directly
    (already flat and batch-offset, as e.g. produced internally by PET's own
    preprocessing), or ``systems`` and ``nl_options`` must be provided so that they can
    be computed by reading each system's own neighbor list.

    :param sample_labels: Labels for all atoms in the batch, with dimensions
        ``["system", "atom"]``, as returned by :func:`get_per_atom_sample_labels`.
    :param centers: Flat tensor of center atom global indices, shape ``(n_edges,)``.
    :param neighbors: Flat tensor of neighbor atom global indices, shape ``(n_edges,)``.
    :param cell_shifts: Integer cell shift vectors for each edge, shape ``(n_edges,
        3)``.
    :param systems: List of systems in the batch. Only used if ``centers``,
        ``neighbors`` and ``cell_shifts`` are not provided.
    :param nl_options: Options for the neighbor list used to enumerate edges. Only used
        if ``centers``, ``neighbors`` and ``cell_shifts`` are not provided.
    :return: Labels with columns ``["system", "first_atom", "second_atom",
        "cell_shift_a", "cell_shift_b", "cell_shift_c"]``, shape ``(n_edges, 6)``.
    """
    if centers is None or neighbors is None or cell_shifts is None:
        if systems is None or nl_options is None:
            raise ValueError(
                "either `centers`, `neighbors` and `cell_shifts` must all be "
                "provided, or `systems` and `nl_options` must be provided so "
                "they can be computed from the systems' neighbor lists"
            )
        centers, neighbors, cell_shifts = _pair_arrays_from_neighbor_lists(
            systems, nl_options
        )

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
        assume_unique=True,
    )


def _pair_arrays_from_neighbor_lists(
    systems: List[System],
    nl_options: NeighborListOptions,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reads each system's own neighbor list and returns the flat, batch-offset
    ``centers``, ``neighbors`` and ``cell_shifts`` tensors required by
    :func:`get_pair_sample_labels`.

    :param systems: List of systems in the batch.
    :param nl_options: Options for the neighbor list used to enumerate edges.
    :return: Tuple of ``(centers, neighbors, cell_shifts)``.
    """
    device = systems[0].positions.device
    nl_values_list: List[torch.Tensor] = []
    num_edges: List[int] = []
    node_offsets_list: List[int] = []

    node_counter = 0
    for system in systems:
        assert len(system.known_neighbor_lists()) >= 1, "no neighbor list found"
        neighbor_list = system.get_neighbor_list(nl_options)
        nl_values = neighbor_list.samples.values
        nl_values_list.append(nl_values)

        system_size = len(system)
        node_offsets_list.append(node_counter)
        num_edges.append(nl_values.shape[0])
        node_counter += system_size

    nl_values = torch.cat(nl_values_list)
    centers = nl_values[:, 0]
    neighbors = nl_values[:, 1]
    cell_shifts = nl_values[:, 2:]

    # Compute the offsets
    total_edges = sum(num_edges)
    num_edges_tensor = torch.tensor(num_edges, device=device, dtype=torch.long)
    node_offsets = torch.tensor(node_offsets_list, device=device, dtype=torch.long)
    edge_offsets = torch.repeat_interleave(
        node_offsets, num_edges_tensor, output_size=total_edges
    ).to(dtype=centers.dtype)

    # Offset the centers and neighbors
    centers = centers + edge_offsets
    neighbors = neighbors + edge_offsets

    return centers, neighbors, cell_shifts
