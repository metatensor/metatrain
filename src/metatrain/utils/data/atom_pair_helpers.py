from typing import Callable, Dict, List, Optional, Tuple

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
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


def get_single_direction_edges(tmap: TensorMap) -> TensorMap:
    """Takes a TensorMap with edges in both directions and returns a
    TensorMap with only one direction of the edges.

    It keeps only:

    - If atom types are different, the edges where the first atom type
      is smaller than the second atom type.
    - If atom types are the same, the edges where the first atom index
      is smaller than the second atom index.

    This function only works for atomic basis data for now.

    :param tmap: A TensorMap containing edge data in both directions.
    :return: A TensorMap containing edge data in only one direction.
    """
    is_atomic_basis = "first_atom_type" in tmap.keys.names
    if not is_atomic_basis:
        raise ValueError(
            "Getting single direction edges is only supported for "
            "atomic basis data for now."
        )

    new_blocks = []
    new_keys = []
    for key, block in tmap.items():
        # If atom types are different, drop blocks where the first
        # atom type is greater than the second atom type, and keep
        # the block untouched when the first atom type is smaller.
        # (this is the easiest case)
        if key["first_atom_type"] > key["second_atom_type"]:
            continue
        elif key["first_atom_type"] < key["second_atom_type"]:
            new_blocks.append(block)
            new_keys.append(key)
            continue
        else:
            # Otherwise, get the edges where the first atom index is smaller
            # than the second one.
            mask = block.samples["first_atom"] < block.samples["second_atom"]
            new_block = TensorBlock(
                values=block.values[mask],
                samples=Labels(
                    names=block.samples.names,
                    values=block.samples.values[mask],
                ),
                components=block.components,
                properties=Labels(
                    names=block.properties.names, values=block.properties.values
                ),
            )
            new_blocks.append(new_block)

            new_keys.append(key)

    return TensorMap(
        blocks=new_blocks,
        keys=Labels(
            names=tmap.keys.names, values=torch.tensor(new_keys, device=tmap.device)
        ),
    )


def get_bidirectional_edges(tmap: TensorMap) -> TensorMap:
    """Takes a TensorMap with edges in only one direction and returns a
    TensorMap with edges in both directions.

    This function only supports atomic basis data that comes from a coupled
    product for now.

    :param tmap: A TensorMap containing edge data in only one direction.
    :return: A TensorMap containing edge data in both directions.
    """
    is_atomic_basis = "first_atom_type" in tmap.keys.names
    is_coupled = "n_1" in tmap.block(0).properties.names

    if not is_atomic_basis or not is_coupled:
        raise ValueError(
            "Getting multi direction edges is only supported for "
            "atomic basis data coming from a coupled product for now."
        )

    # Get the indices of keys, samples and properties fields so that
    # we can permute them.
    i_first = tmap.block(0).samples.names.index("first_atom")
    i_second = tmap.block(0).samples.names.index("second_atom")
    cell_shift_a = tmap.block(0).samples.names.index("cell_shift_a")
    cell_shift_b = tmap.block(0).samples.names.index("cell_shift_b")
    cell_shift_c = tmap.block(0).samples.names.index("cell_shift_c")
    if is_coupled:
        i_n1 = tmap.block(0).properties.names.index("n_1")
        i_n2 = tmap.block(0).properties.names.index("n_2")
        i_l1 = tmap.block(0).properties.names.index("l_1")
        i_l2 = tmap.block(0).properties.names.index("l_2")
    if is_atomic_basis:
        i_type1 = tmap.keys.names.index("first_atom_type")
        i_type2 = tmap.keys.names.index("second_atom_type")

    new_blocks = []
    new_keys = []
    for key, block in tmap.items():
        if is_atomic_basis:
            # If the block corresponds to edges with different atom types,
            # we keep the block untouched, and we will also create the
            # reverse block.
            if key["first_atom_type"] < key["second_atom_type"]:
                new_blocks.append(block)
                new_keys.append(key.values)
            elif key["first_atom_type"] > key["second_atom_type"]:
                raise ValueError(
                    "Expected the input TensorMap to only contain one direction "
                    "of the edges."
                )

        # Get the reverse connections (i -> j becomes j -> i, and the supercell
        # shift is reversed).
        reverse_samples = block.samples.values.clone()
        reverse_samples[:, [i_first, i_second]] = reverse_samples[
            :, [i_second, i_first]
        ]
        reverse_samples[:, [cell_shift_a, cell_shift_b, cell_shift_c]] *= -1

        # Get the values for the data of the reverse connections.
        reverse_values = block.values
        if is_coupled:
            # If o3_sigma is -1, the reverse block should have its values negated.
            reverse_values = reverse_values * key["o3_sigma"]

        # Get the properties of the reverse block.
        properties = block.properties.values.clone()
        if is_coupled:
            # Swap n_1 with n_2, and l_1 with l_2.
            properties[:, [i_n1, i_n2, i_l1, i_l2]] = properties[
                :, [i_n2, i_n1, i_l2, i_l1]
            ]
        reverse_properties = Labels(names=block.properties.names, values=properties)

        # Now we can construct the final block.
        if is_atomic_basis and key["first_atom_type"] < key["second_atom_type"]:
            # Block with only the reverse connections.
            # This is because we are creating the block with first_atom_type greater
            # than second_atom_type (the opposite one we already have it, see above).
            new_block = TensorBlock(
                values=reverse_values,
                samples=Labels(
                    names=block.samples.names,
                    values=reverse_samples,
                ),
                components=block.components,
                properties=reverse_properties,
            )
            new_key = key.values.clone()
            new_key[[i_type1, i_type2]] = new_key[[i_type2, i_type1]]
        else:
            # Block containing both connections.
            selection = block.properties.select(reverse_properties)
            new_block = TensorBlock(
                values=torch.cat([block.values, reverse_values[..., selection]], dim=0),
                samples=Labels(
                    names=block.samples.names,
                    values=torch.cat([block.samples.values, reverse_samples], dim=0),
                ),
                components=block.components,
                properties=block.properties,
            )
            new_key = key.values

        new_blocks.append(new_block)
        new_keys.append(new_key)

    return TensorMap(
        blocks=new_blocks,
        keys=Labels(names=tmap.keys.names, values=torch.stack(new_keys)),
    )


def get_bidirectional_edges_transform(
    target_info_dict: dict[str, TargetInfo],
    extra_data_info_dict: dict[str, TargetInfo],
) -> tuple[Callable, Callable]:
    """
    Get transform functions to go from single direction edges to bidirectional
    edges and the reverse.

    :param target_info_dict: Dictionary mapping target names to TargetInfo objects.
    :param extra_data_info_dict: Dictionary mapping extra data names to TargetInfo
        objects.

    :return: Two functions: the first one transforms single direction edges to
      bidirectional and the second one transforms bidirectional edges to
      single direction.
    """

    def transform(
        systems: list[System],
        targets: dict[str, TensorMap],
        extra: dict[str, TensorMap],
    ) -> tuple[list[System], dict[str, TensorMap], dict[str, TensorMap]]:
        """
        Transform function that gets the bidirectional edges from the
        single direction ones.

        :param systems: List of systems.
        :param targets: Dictionary containing the targets corresponding to the systems.
        :param extra: Dictionary containing any extra data.
        :return: The systems, targets and extra data with bidirectional data
           for the edges.
        """
        for name, tensor in targets.items():
            if (
                name in target_info_dict
                and target_info_dict[name].sample_kind == "atom_pair"
            ):
                targets[name] = get_bidirectional_edges(tensor)

        for name, tensor in extra.items():
            if (
                name in extra_data_info_dict
                and extra_data_info_dict[name].sample_kind == "atom_pair"
            ):
                targets[name] = get_bidirectional_edges(tensor)

        return systems, targets, extra

    def reverse_transform(
        systems: list[System],
        targets: dict[str, TensorMap],
        extra: dict[str, TensorMap],
    ) -> tuple[list[System], dict[str, TensorMap], dict[str, TensorMap]]:
        """
        Transform function that gets the single direction edges from the
        bidirectional ones.

        :param systems: List of systems.
        :param targets: Dictionary containing the targets corresponding to the systems.
        :param extra: Dictionary containing any extra data.
        :return: The systems, targets and extra data with data on only one direction
          of the edges.
        """
        for name, tensor in targets.items():
            if (
                name in target_info_dict
                and target_info_dict[name].sample_kind == "atom_pair"
            ):
                targets[name] = get_single_direction_edges(tensor)

        for name, tensor in extra.items():
            if (
                name in extra_data_info_dict
                and extra_data_info_dict[name].sample_kind == "atom_pair"
            ):
                targets[name] = get_single_direction_edges(tensor)

        return systems, targets, extra

    return transform, reverse_transform
