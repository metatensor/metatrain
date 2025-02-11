from typing import List, Tuple, Optional

import numpy as np
import torch
import vesin

import metatensor.torch as mts
from metatensor.torch import Labels, TensorBlock, TensorMap


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
    values = block.values  # .clone() # TODO: is this to be cloned?

    # Find the indices of the samples to symmetrize
    idx_to_symmetrize = all_samples.select(
        Labels(
            names=sample_names,
            values=permuted_samples,
        )
    )

    # print(sample_names, permuted_samples)
    # print(idx_to_symmetrize.shape, values.shape, idx_to_symmetrize)
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
        samples=reduced_samples,  # mts.Labels(b.samples.names, np.array(samples)),
        values=values_plus,
        components=block.components,
        properties=properties,
    )
    block_minus = TensorBlock(
        samples=reduced_samples,  # mts.Labels(b.samples.names, np.array(samples)),
        values=values_minus,
        components=block.components,
        properties=properties,
    )

    return block_plus, block_minus


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

    return in_keys_edge_sliced, out_properties_edge_sliced


def get_neighbor_list(
    frames,
    frame_idxs: List[int],
    cutoff: float,
) -> mts.Labels:
    """
    Computes the neighbour list for each frame in ``frames`` and returns a
    :py:class:`metatensor
    """
    nl = vesin.NeighborList(cutoff=cutoff, full_list=True)

    labels_values = []
    for A, frame in zip(frame_idxs, frames):

        # Compute the neighbor list
        if np.any([d == 0 for d in frame.cell]):  # for ase
            # if np.any([d == 0 for d in frame.cell]):  # for chemfiles
            box = np.zeros((3, 3))
            periodic = False
        else:
            box = frame.cell.matrix
            periodic = True

        i_list, j_list, S_list = nl.compute(
            points=frame.positions,
            box=box,
            periodic=periodic,
            quantities="ijS",
        )

        # Now add in the self terms as vesin does not include them
        i_list = np.concatenate([i_list, np.arange(len(frame.positions), dtype=int)])
        j_list = np.concatenate([j_list, np.arange(len(frame.positions), dtype=int)])
        S_list = np.concatenate(
            [S_list, np.array([0, 0, 0] * len(frame.positions)).reshape(-1, 3)],
            dtype=int,
        )

        # Add dimension for system index
        for i, j, S in zip(i_list, j_list, S_list):
            a, b, c = S
            label = [A, i, j, a, b, c]
            if label not in labels_values:
                labels_values.append(label)

    return mts.Labels(
        names=[
            "system",
            "first_atom",
            "second_atom",
            "cell_shift_a",
            "cell_shift_b",
            "cell_shift_c",
        ],
        values=torch.tensor(labels_values, dtype=torch.int32),
    )
