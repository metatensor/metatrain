from typing import List, Tuple

import torch
from metatensor.torch import Labels, TensorMap

from metatomic.torch import System, NeighborListOptions


def extract_key_value(key_str: str, dimension_name: str) -> int:
    idx = key_str.find(dimension_name + '_')
    if idx == -1:
        raise KeyError(f"Dimension '{dimension_name}' not found in key string.")
    
    # Start after the matched dimension_name and underscore
    start = idx + len(dimension_name) + 1
    end = start
    while end < len(key_str) and (key_str[end] == '-' or key_str[end].isdigit()):
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


def get_edge_sample_labels(
    systems: List[System],
    node_sample_labels: Labels,
    nl_options: NeighborListOptions,
    device: torch.device,
) -> List[Labels]:
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
                node_sample_labels.values[:, 1]. unsqueeze(1),  # i == j
                torch.zeros(  # cell shifts are 0
                    (node_sample_labels.values.shape[0], 3),
                    dtype=node_sample_labels.values.dtype,
                    device=node_sample_labels.values.device,
                ),
            ]
        )
    )

    edge_sample_values_2_center = []
    for system_idx, system in enumerate(systems):
        neighbor_list = system.get_neighbor_list(nl_options)
        nl_values = neighbor_list.samples.values

        edge_sample_values_2_center.append(
            torch.hstack(
                [torch.full((nl_values.shape[0], 1), system_idx), nl_values],
            )
        )
    edge_sample_labels_2_center = Labels(
        sample_names,
        torch.vstack(edge_sample_values_2_center),
    )

    return [
        edge_sample_labels_1_center, edge_sample_labels_2_center
    ]


def get_permutation_symmetrization_arrays(
    systems: List[System],
    edge_sample_labels_2_center: Labels,
) -> List[torch.Tensor]:
    
    assert edge_sample_labels_2_center.names == [
        "system",
        "first_atom",
        "second_atom",
        "cell_shift_a",
        "cell_shift_b",
        "cell_shift_c",
    ]

    # Get the atom types
    atom_types = []
    for sample in edge_sample_labels_2_center.values:
        system_idx = sample[0]
        first_atom_idx = sample[1]
        second_atom_idx = sample[2]
        atom_types.append(
            [
                systems[system_idx].types[first_atom_idx],
                systems[system_idx].types[second_atom_idx],
            ]
        )
    atom_types = torch.tensor(atom_types, dtype=torch.int32, device=edge_sample_labels_2_center.device)
    
    # build the masks for same and different atom types
    samples_mask_2_center_same_types = atom_types[:, 0] == atom_types[:, 1]
    samples_mask_2_center_diff_types = atom_types[:, 0] != atom_types[:, 1]

    # build the samples labels for atom pairs with the same and different atom types
    edge_sample_labels_2_center_same_types = Labels(
        edge_sample_labels_2_center.names,
        edge_sample_labels_2_center.values[samples_mask_2_center_same_types],
    )
    edge_sample_labels_2_center_diff_types = Labels(
        edge_sample_labels_2_center.names,
        edge_sample_labels_2_center.values[samples_mask_2_center_diff_types]
    )

    # create permuted sample labels by swapping the atom indices and inverting the sign
    # of the cell shifts
    edge_sample_values_2_center_same_types_perm = edge_sample_labels_2_center_same_types.permute(
        [0, 2, 1, 3, 4, 5]
    ).values.clone()
    edge_sample_values_2_center_same_types_perm[:, 3:6] *= -1
    edge_sample_labels_2_center_same_types_perm = Labels(
        edge_sample_labels_2_center_same_types.names,
        edge_sample_values_2_center_same_types_perm,
    )

    # find the map from the original edge samples to the permuted samples
    permuted_samples_map_same_types = edge_sample_labels_2_center_same_types.select(
        edge_sample_labels_2_center_same_types_perm
    )

    return [
        samples_mask_2_center_same_types,
        samples_mask_2_center_diff_types,
        permuted_samples_map_same_types,
        edge_sample_labels_2_center_same_types,
        edge_sample_labels_2_center_diff_types,
    ]

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
