from typing import Dict, List, Tuple

import torch
from metatensor.torch import Labels, TensorBlock
from metatomic.torch import NeighborListOptions, System


def pad_block(
    block: TensorBlock,
    axis: str,
    padded_labels: Labels,
    pad_value: float = torch.nan,
) -> TensorBlock:
    """
    Takes a TensorBlock and pads it along the specified axis with the ``pad_value``,
    using the provided padded_labels to determine the metadata structure of the new
    block.

    Any of the samples/properties that are already present in the block will be filled
    with the original values, while the new samples/properties will be filled with
    zeros.

    :param block: The TensorBlock to pad.
    :param axis: The axis to pad, either "samples" or "properties".
    :param padded_labels: The Labels object containing the desired samples/properties
        after padding.
    :param pad_value: The value to use for padding. Default is torch.nan.
    :return: The padded TensorBlock.
    """

    def get_idxs_pad_idxs_original(
        padded_labels: Labels, original_labels: Labels
    ) -> Tuple[torch.tensor, torch.tensor]:
        # First find the intersection
        intersection = padded_labels.intersection(original_labels)
        # Get the indices of the intersection in the padded_labels
        nonzero_idxs_padded = padded_labels.select(intersection)
        # Get the indices of the intersection in the original_labels
        nonzero_idxs_original = original_labels.select(intersection)

        return nonzero_idxs_padded, nonzero_idxs_original

    if axis == "samples":
        samples = padded_labels
        properties = block.properties

        block_values = (
            torch.ones(
                (len(samples), *[len(c) for c in block.components], len(properties)),
                dtype=block.values.dtype,
                device=block.values.device,
            )
            * pad_value
        )
        nonzero_idxs_padded, nonzero_idxs_original = get_idxs_pad_idxs_original(
            samples, block.samples
        )
        assert len(nonzero_idxs_padded) == len(nonzero_idxs_original)
        if len(nonzero_idxs_original) > 0:
            block_values[nonzero_idxs_padded] = block.values[nonzero_idxs_original]

    else:
        assert axis == "properties"
        samples = block.samples
        properties = padded_labels

        block_values = (
            torch.ones(
                (len(samples), *[len(c) for c in block.components], len(properties)),
                dtype=block.values.dtype,
                device=block.values.device,
            )
            * pad_value
        )
        nonzero_idxs_padded, nonzero_idxs_original = get_idxs_pad_idxs_original(
            properties, block.properties
        )
        assert len(nonzero_idxs_padded) == len(nonzero_idxs_original)
        if len(nonzero_idxs_original) > 0:
            block_values[..., nonzero_idxs_padded] = block.values[
                ..., nonzero_idxs_original
            ]

    return TensorBlock(
        samples=samples,
        components=block.components,
        properties=properties,
        values=block_values,
    )


def get_atom_sample_labels(systems: List[System], device: torch.device) -> Labels:
    """
    Builds the atom sample labels for the input ``systems``.
    :param systems: List of systems to build the atom sample labels for.
    :param device: The device to put the labels on.
    :return: The atom sample labels.
    """
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
    return node_sample_labels


def get_pair_sample_labels_onsite(
    systems: List[System],
    sample_labels: Labels,
    device: torch.device,
) -> Labels:
    """
    Builds the onsite pair samples labels for the input ``systems`` by adding
    "second_atom" and "cell_shift_x" dimensions to the atom sample labels.

    :param systems: List of systems to build the pair sample labels for.
    :param sample_labels: The sample labels for per-atom quantities.
    :param device: The device to put the labels on.
    :return: A dictionary with the pair sample labels for the onsite and offsite blocks.
    """
    sample_names = [
        "system",
        "first_atom",
        "second_atom",
        "cell_shift_a",
        "cell_shift_b",
        "cell_shift_c",
    ]
    # Onsite labels
    pair_sample_labels_onsite = Labels(
        sample_names,
        torch.hstack(
            [
                sample_labels.values,
                sample_labels.values[:, 1].unsqueeze(1),  # i == j
                torch.zeros(  # cell shifts are 0
                    (sample_labels.values.shape[0], 3),
                    dtype=torch.int32,
                    device=device,
                ),
            ]
        ),
    )

    return pair_sample_labels_onsite


def get_pair_sample_labels(
    systems: List[System],
    sample_labels: Labels,
    nl_options: NeighborListOptions,
    device: torch.device,
) -> Dict[str, Labels]:
    """
    Builds the pair samples labels for the input ``systems``, based on the pre-computed
    neighbor list. Returns the labels for both the onsite (``n_centers=1``) and offsite
    (``n_centers=2``) blocks, in a dictionary.

    :param systems: List of systems to build the pair sample labels for.
    :param sample_labels: The sample labels for per-atom quantities.
    :param nl_options: The neighbor list options to use for building the offsite labels.
    :param device: The device to put the labels on.
    :return: A dictionary with the pair sample labels for the onsite and offsite blocks.
    """
    sample_names = [
        "system",
        "first_atom",
        "second_atom",
        "cell_shift_a",
        "cell_shift_b",
        "cell_shift_c",
    ]
    # Onsite labels
    pair_sample_labels_onsite = Labels(
        sample_names,
        torch.hstack(
            [
                sample_labels.values,
                sample_labels.values[:, 1].unsqueeze(1),  # i == j
                torch.zeros(  # cell shifts are 0
                    (sample_labels.values.shape[0], 3),
                    dtype=torch.int32,
                    device=device,
                ),
            ]
        ),
    )

    # Offsite labels
    pair_sample_values_offsite = []
    for system_idx, system in enumerate(systems):
        neighbor_list = system.get_neighbor_list(nl_options)
        nl_values = neighbor_list.samples.values

        pair_sample_values_offsite.append(
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
    pair_sample_values_offsite = torch.vstack(pair_sample_values_offsite)

    # Create the labels for the edge samples
    pair_sample_labels_offsite = Labels(sample_names, pair_sample_values_offsite).to(
        device=device
    )

    return {"onsite": pair_sample_labels_onsite, "offsite": pair_sample_labels_offsite}
