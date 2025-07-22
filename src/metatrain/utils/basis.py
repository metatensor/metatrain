from typing import List

import torch
from metatensor.torch import Labels
from metatomic.torch import NeighborListOptions, System


def extract_key_value(key_str: str, dimension_name: str) -> int:
    idx = key_str.find(dimension_name + "_")
    if idx == -1:
        raise KeyError(f"Dimension '{dimension_name}' not found in key string.")

    # Start after the matched dimension_name and underscore
    start = idx + len(dimension_name) + 1
    end = start
    while end < len(key_str) and (key_str[end] == "-" or key_str[end].isdigit()):
        end += 1

    value_str = key_str[start:end]
    return int(value_str)


def get_system_indices_and_node_sample_labels(
    systems: List[System], device: torch.device
):
    system_indices = torch.concatenate(
        [
            torch.full(
                (len(system),),
                i_system,
                device=device,
            )
            for i_system, system in enumerate(systems)
        ],
    )

    sample_values = torch.stack(
        [
            system_indices,
            torch.concatenate(
                [
                    torch.arange(
                        len(system),
                        device=device,
                    )
                    for system in systems
                ],
            ),
        ],
        dim=1,
    )
    node_sample_labels = Labels(
        names=["system", "atom"],
        values=sample_values,
    )
    return system_indices, node_sample_labels


def get_edge_sample_labels_1_center(
    node_sample_labels: Labels,
    device: torch.device,
) -> Labels:
    """
    Builds the edge samples labels for the input ``systems``, based on the pre-computed
    neighbor list. Returns the labels for both the ``n_centers=1`` and ``n_centers=2``
    blocks.
    """
    sample_names = [
        "system",
        "first_atom",
        "second_atom",
        "cell_shift_a",
        "cell_shift_b",
        "cell_shift_c",
    ]
    # first build the on-site (n_centers=1) labels from the standard node labels
    edge_sample_labels_1_center = Labels(
        sample_names,
        torch.hstack(
            [
                node_sample_labels.values,
                node_sample_labels.values[:, 1].unsqueeze(1),  # i == j
                torch.zeros(  # cell shifts are 0
                    (node_sample_labels.values.shape[0], 3),
                    dtype=torch.int32,
                    device=device,
                ),
            ]
        ),
    )
    return edge_sample_labels_1_center


def get_edge_sample_labels_2_center(
    systems: List[System],
    nl_options: NeighborListOptions,
    device: torch.device,
) -> Labels:
    """
    Builds the edge samples labels for the input ``systems``, based on the pre-computed
    neighbor list. Returns the labels for both the ``n_centers=1`` and ``n_centers=2``
    blocks.
    """
    sample_names = [
        "system",
        "first_atom",
        "second_atom",
        "cell_shift_a",
        "cell_shift_b",
        "cell_shift_c",
    ]
    edge_sample_values_2_center = []
    for system_idx, system in enumerate(systems):
        neighbor_list = system.get_neighbor_list(nl_options)
        nl_values = neighbor_list.samples.values

        edge_sample_values_2_center.append(
            torch.hstack(
                [
                    torch.full(
                        (nl_values.shape[0], 1),
                        system_idx,
                        dtype=torch.int32,
                        device=device,
                    ),
                    nl_values,
                ],
            )
        )
    edge_sample_labels_2_center = Labels(
        sample_names,
        torch.vstack(edge_sample_values_2_center),
    ).to(device=device)
    return edge_sample_labels_2_center


def get_permutation_symmetrization_arrays(
    systems: List[System],
    edge_sample_labels_2_center: Labels,
):
    assert edge_sample_labels_2_center.names == [
        "system",
        "first_atom",
        "second_atom",
        "cell_shift_a",
        "cell_shift_b",
        "cell_shift_c",
    ]

    # If we have no offsite terms, return empty arrays
    if len(edge_sample_labels_2_center.values) == 0:
        device = edge_sample_labels_2_center.values.device
        return (
            torch.tensor([], dtype=torch.bool, device=device),
            torch.tensor([], dtype=torch.bool, device=device),
            torch.tensor([], dtype=torch.int32, device=device),
            Labels(
                edge_sample_labels_2_center.names,
                torch.tensor([], dtype=torch.int32, device=device).reshape(0, 6),
            ),
            Labels(
                edge_sample_labels_2_center.names,
                torch.tensor([], dtype=torch.int32, device=device).reshape(0, 6),
            ),
        )

    # Get the atom types
    atom_types = torch.vstack(
        [
            systems[sample[0]].types[sample[1:3]]
            for sample in edge_sample_labels_2_center.values
        ]
    )

    # build the masks for same and different atom types
    samples_mask_2_center_same_types: torch.Tensor = (
        atom_types[:, 0] == atom_types[:, 1]
    )
    samples_mask_2_center_diff_types: torch.Tensor = (
        atom_types[:, 0] != atom_types[:, 1]
    )

    # build the samples labels for atom pairs with the same and different atom types
    edge_sample_labels_2_center_same_types: Labels = Labels(
        edge_sample_labels_2_center.names,
        edge_sample_labels_2_center.values[samples_mask_2_center_same_types],
    )
    edge_sample_labels_2_center_diff_types: Labels = Labels(
        edge_sample_labels_2_center.names,
        edge_sample_labels_2_center.values[samples_mask_2_center_diff_types],
    )

    # create permuted sample labels by swapping the atom indices and inverting the sign
    # of the cell shifts
    edge_sample_values_2_center_same_types_perm: torch.Tensor = (
        edge_sample_labels_2_center_same_types.permute(
            [0, 2, 1, 3, 4, 5]
        ).values.clone()
    )
    edge_sample_values_2_center_same_types_perm[:, 3:6] *= -1
    edge_sample_labels_2_center_same_types_perm = Labels(
        edge_sample_labels_2_center_same_types.names,
        edge_sample_values_2_center_same_types_perm,
    )

    # find the map from the original edge samples to the permuted samples
    permuted_samples_map_same_types: torch.Tensor = (
        edge_sample_labels_2_center_same_types.select(
            edge_sample_labels_2_center_same_types_perm
        )
    )

    return (
        samples_mask_2_center_same_types,
        samples_mask_2_center_diff_types,
        permuted_samples_map_same_types,
        edge_sample_labels_2_center_same_types,
        edge_sample_labels_2_center_diff_types,
    )


def get_sample_labels_block(
    key: str,
    sample_kind: str,
    node_sample_labels: Labels,
    edge_sample_labels_1_center: Labels,
    edge_sample_labels_2_center: Labels,
    edge_sample_labels_2_center_same_types: Labels,
    edge_sample_labels_2_center_diff_types: Labels,
) -> Labels:
    """Returns the correct block samples labels for the given
    output, based on the key"""
    if "n_centers" not in key:
        sample_labels_block = node_sample_labels
    else:
        if extract_key_value(key, "n_centers") == 1:
            if sample_kind == "per_atom":
                sample_labels_block = node_sample_labels
            else:
                assert sample_kind == "per_pair"
                sample_labels_block = edge_sample_labels_1_center

        else:
            assert sample_kind == "per_pair"
            if "s2_pi" in key:
                s2_pi = extract_key_value(key, "s2_pi")
                assert s2_pi in [0, 1, -1]
                if s2_pi == 0:
                    sample_labels_block = edge_sample_labels_2_center_diff_types
                else:
                    sample_labels_block = edge_sample_labels_2_center_same_types
            else:
                sample_labels_block = edge_sample_labels_2_center

    return sample_labels_block
