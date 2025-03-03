from typing import List, Optional, Tuple

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatrain.experimental.nanopet.modules.augmentation import (
    get_random_inversion,
    get_random_rotation,
)

def get_node_layout(in_keys_node: Labels, out_properties_node: Labels) -> TensorMap:
    """
    Given the keys and out properties of a node target, returns the "layout" TensorMap
    needed to create a `DatasetInfo` object for use in metatrain.
    """
    return TensorMap(
        in_keys_node,
        [
            TensorBlock(
                samples=Labels.empty(["system", "atom"]),
                components=[
                    Labels(
                        "o3_mu",
                        torch.arange(
                            -k["o3_lambda"], k["o3_lambda"] + 1
                        ).reshape(-1, 1),
                    )
                ],
                properties=out_props,
                values=torch.empty(
                    0,
                    2 * k["o3_lambda"] + 1,
                    len(out_props),
                ),
            )
            for k, out_props in zip(in_keys_node, out_properties_node)
        ],
    )

def get_edge_layout(in_keys_edge: Labels, out_properties_edge: Labels) -> TensorMap:
    """
    Given the keys and out properties of an edge target, returns the "layout" TensorMap
    needed to create a `DatasetInfo` object for use in metatrain.
    """
    return TensorMap(
        in_keys_edge,
        [
            TensorBlock(
                samples=Labels.empty(
                    [
                        "system",
                        "first_atom",
                        "second_atom",
                        "cell_shift_a",
                        "cell_shift_b",
                        "cell_shift_c"
                    ]
                ),
                components=[
                    Labels(
                        "o3_mu",
                        torch.arange(
                            -k["o3_lambda"], k["o3_lambda"] + 1
                        ).reshape(-1, 1),
                    )
                ],
                properties=out_props,
                values=torch.empty(
                    0,
                    2 * k["o3_lambda"] + 1,
                    len(out_props),
                ),
            )
            for k, out_props in zip(
                in_keys_edge, out_properties_edge
            )
        ],
    )

def get_system_transformations(systems) -> List[torch.Tensor]:
    """
    Returns a series of random transformations to be applied for each system in
    ``systems``.
    """
    rotations = [get_random_rotation() for _ in range(len(systems))]
    inversions = [get_random_inversion() for _ in range(len(systems))]
    return rotations, inversions


def keys_triu_center_type(
    in_keys_edge: Labels, out_properties_edge: List[Labels]
) -> TensorMap:
    idxs_to_keep = []
    for key_i, key in enumerate(in_keys_edge):
        # Keep blocks where the first atom type is less than the second atom type
        if key["first_atom_type"] <= key["second_atom_type"]:
            idxs_to_keep.append(key_i)

    in_keys_edge_sliced = Labels(
        in_keys_edge.names,
        in_keys_edge.values[idxs_to_keep],
    )
    out_properties_edge_sliced = [
        out_props
        for i, out_props in enumerate(out_properties_edge)
        if i in idxs_to_keep
    ]

    assert len(in_keys_edge_sliced) == len(out_properties_edge_sliced)

    return in_keys_edge_sliced, out_properties_edge_sliced


def symmetrize_samples(
    block: TensorBlock, second_atom_type: Optional[int] = None
) -> Tuple[TensorBlock]:
    """
    Symmetrizes the samples dimension of a tensor block.
    """

    # Define the samples
    all_samples = block.samples
    sample_names = all_samples.names

    # Permute the samples to get the negative samples
    permuted_samples = all_samples.permute([0, 2, 1, 3, 4, 5]).values.clone()
    permuted_samples[:, -3:] *= -1
    values = block.values  # .clone()  # TODO: is this to be cloned?

    # Find the indices of the samples to symmetrize
    idx_to_symmetrize = all_samples.select(
        Labels(
            names=sample_names,
            values=permuted_samples,
        )
    )

    # Symmetrize the sample values
    values_plus = values + values[idx_to_symmetrize]
    values_minus = values - values[idx_to_symmetrize]

    reduced_samples_mask = (
        (all_samples.values[:, 1] < all_samples.values[:, 2])
        & torch.isclose(
            torch.linalg.norm(1.0 * all_samples.values[:, -3:]), torch.tensor(0.0)
        )
    ) | (
        (all_samples.values[:, 1] <= all_samples.values[:, 2])
        & (
            ~torch.isclose(
                torch.linalg.norm(1.0 * all_samples.values[:, -3:]), torch.tensor(0.0)
            )
        )
    )
    reduced_samples = Labels(
        sample_names,
        all_samples.values[reduced_samples_mask],
    )
    values_plus = values_plus[reduced_samples_mask]
    values_minus = values_minus[reduced_samples_mask]

    if second_atom_type is not None:
        properties = block.properties.insert(
            1,
            "neighbor_2_type",
            torch.tensor(block.properties.values.shape[0] * [second_atom_type]),
        )
    else:
        properties = block.properties

    block_plus = TensorBlock(
        samples=reduced_samples,  # Labels(b.samples.names, np.array(samples)),
        values=values_plus,
        components=block.components,
        properties=properties,
    )
    block_minus = TensorBlock(
        samples=reduced_samples,  # Labels(b.samples.names, np.array(samples)),
        values=values_minus,
        components=block.components,
        properties=properties,
    )

    return block_plus, block_minus


def symmetrize_predictions_node(
    atomic_types: List[int],
    predictions_node: TensorMap,
    in_keys_node: Labels,
    systems,
) -> TensorMap:
    """Symmetrize PET node predictions."""

    # Create a dictionary storing the atomic indices for each center type
    slice_nodes = {center_type: [] for center_type in atomic_types}
    for A, system in enumerate(systems):
        for i, center_type in enumerate(system.types):
            slice_nodes[int(center_type)].append([A, i])

    # Slice the predictions TensorMap to create blocks for the different center types
    # with the correct atomic samples
    node_blocks = []
    for key in in_keys_node:
        center_type = int(key["center_type"])

        block = slice(
            predictions_node,
            "samples",
            Labels(
                ["system", "atom"],
                torch.tensor(slice_nodes[center_type], dtype=torch.int32).reshape(
                    -1, 2
                ),
            ),
        )[0]

        node_blocks.append(block)

    return TensorMap(in_keys_node, node_blocks)


def symmetrize_predictions_edge(
    atomic_types: List[int],
    predictions_edge: TensorMap,
    in_keys_edge: Labels,
    systems,
) -> TensorMap:
    """
    Symmetrize PET edge predictions
    """
    apply_permutational_symmetry = "block_type" in in_keys_edge.names

    slice_edges = {
        (first_atom_type, second_atom_type): []
        for first_atom_type in atomic_types
        for second_atom_type in atomic_types
    }
    for A, system in enumerate(systems):
        for i, first_atom_type in enumerate(system.types):
            for j, second_atom_type in enumerate(system.types):
                slice_edges[(int(first_atom_type), int(second_atom_type))].append(
                    [A, i, j]
                )

    # Edges (properly symmetrized)
    edge_blocks = []
    for key in in_keys_edge:
        Z1 = int(key["first_atom_type"])
        Z2 = int(key["second_atom_type"])

        # Slice to the relevant types, which could leave a block with zero samples
        block = slice(
            predictions_edge,
            "samples",
            Labels(
                ["system", "first_atom", "second_atom"],
                torch.tensor(slice_edges[(Z1, Z2)], dtype=torch.int32).reshape(-1, 3),
            ),
        )[0]

        # Symmetrize
        if Z1 == Z2 and apply_permutational_symmetry:
            block_plus, block_minus = symmetrize_samples(block)
            if key["block_type"] == 1:
                edge_blocks.append(block_plus)
            elif key["block_type"] == -1:
                edge_blocks.append(block_minus)
            else:
                raise ValueError(f"Block type must be 1 or -1 for Z1=Z2={Z1}")
        else:
            edge_blocks.append(block)

    return TensorMap(in_keys_edge, edge_blocks)


def reindex_tensormap(
    tensor: TensorMap,
    system_ids: List[int],
) -> TensorMap:
    """
    Takes a single TensorMap `tensor` containing data on multiple systems and re-indexes
    the "system" dimension of the samples. Assumes input has numeric system indices from
    {0, ..., N_system - 1} (inclusive), and maps these indices one-to-one with those
    passed in ``system_ids``.
    """
    assert tensor.sample_names[0] == "system"

    index_mapping = {i: A for i, A in enumerate(system_ids)}

    def new_row(row):
        return [index_mapping[row[0].item()]] + [i for i in row[1:]]

    new_blocks = []
    for block in tensor.blocks():
        new_samples = Labels(
            names=block.samples.names,
            values=torch.tensor(
                [new_row(row) for row in block.samples.values],
                dtype=torch.int32,
            ).reshape(-1, len(block.samples.names)),
        )
        new_block = TensorBlock(
            values=block.values,
            samples=new_samples,
            components=block.components,
            properties=block.properties,
        )
        new_blocks.append(new_block)

    return TensorMap(tensor.keys, new_blocks)


def add_back_invariant_mean(tensor: TensorMap, mean_tensor: TensorMap) -> TensorMap:
    """
    Adds back the mean to the invariant blocks of the input ``tensor`` using the
    ``mean_tensor`` layer.
    """
    for key, mean_block in mean_tensor.items():
        if key not in tensor.keys:
            continue
        tensor_block = tensor[key]
        tensor_block.values[:] += mean_block.values

    return tensor


def revert_standardization(tensor: TensorMap, standardizer: TensorMap) -> TensorMap:
    """
    Un-standardizes the input ``tensor`` using the ``standardizer`` layer.
    """
    for key, block in tensor.items():
        block.values[:] *= standardizer.block(key).values

    return tensor
