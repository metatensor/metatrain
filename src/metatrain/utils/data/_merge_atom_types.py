from typing import Callable

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from .target_info import TargetInfo


def merge_types(tensor_map: TensorMap, fill_value: float = torch.nan) -> TensorMap:
    """Merge the blocks of a `TensorMap` that differ only in the atom types.

    The merging is done by padding the blocks with the ``fill_value`` so that they
    all have the same shape, and then concatenating them along the sample dimension.

    :param tensor_map: The `TensorMap` for which to merge the blocks.
    :param fill_value: The value to use for padding the blocks.
    :return: A new `TensorMap` with the merged blocks.
    """
    # From the keys that are not related to atom types,
    # get the unique values. These will be the keys of the
    # new tensormap.
    non_type_indices = [
        i
        for i, name in enumerate(tensor_map.keys.names)
        if not name.endswith("atom_type")
    ]
    new_keys_names = [tensor_map.keys.names[i] for i in non_type_indices]
    new_keys = tensor_map.keys.values[:, non_type_indices].unique(dim=0)

    # Build the new blocks for each key.
    new_blocks = []
    for key_vals in new_keys:
        # Search for all atom type blocks that have these key values.
        all_blocks = tensor_map.blocks(
            {
                tensor_map.keys.names[i_name]: int(key_vals[i])
                for i, i_name in enumerate(non_type_indices)
            }
        )

        # Get their shapes and determine the shape of the new merged block.
        # Initialize the new block with the fill value.
        all_shapes = torch.tensor([block.values.shape for block in all_blocks])
        padded_shape = (
            sum(all_shapes[:, 0]),
            *all_shapes[0, 1:-1],
            max(all_shapes[:, -1]),
        )
        padded_values = torch.full(
            padded_shape, fill_value, dtype=all_blocks[0].values.dtype
        )

        # Move all blocks' values to the new padded block.
        i_sample = 0
        for block in all_blocks:
            n_samples, *n_components, n_properties = block.values.shape
            padded_values[i_sample : i_sample + n_samples, ..., :n_properties] = (
                block.values
            )
            i_sample += n_samples

        # With all the values in the new padded block, we can
        # now build the new TensorBlock.
        new_block = TensorBlock(
            values=padded_values,
            samples=Labels(
                names=all_blocks[0].samples.names,
                values=torch.concat(
                    [block.samples.values for block in all_blocks], dim=0
                ),
            ),
            components=all_blocks[0].components,
            properties=Labels(
                all_blocks[0].properties.names,
                all_blocks[torch.argmax(all_shapes[:, -1])].properties.values,
            ),
        )

        new_blocks.append(new_block)

    return TensorMap(
        keys=Labels(names=new_keys_names, values=new_keys),
        blocks=new_blocks,
    )


def get_merge_types_transform(
    target_info_dict: dict[str, TargetInfo],
    extra_data_info_dict: dict[str, TargetInfo],
    fill_value: float = torch.nan,
) -> Callable:
    """Get a transform function that merges blocks that only differ in atom types.

    The transformation is applied on the targets and extra data that have
    ``is_atomic_basis`` set to `True` in the corresponding `TargetInfo` objects.

    See ``merge_types`` for more details on how the merging is done.

    :param target_info_dict: Dictionary mapping target keys to their `TargetInfo`.
    :param extra_data_info_dict: Dictionary mapping extra data keys to their
      `TargetInfo`.
    :param fill_value: The value to use for padding the blocks when merging.
    :return: A transform function that merges blocks with different atom types.
    """

    def transform(
        systems: list[System],
        targets: dict[str, TensorMap],
        extra: dict[str, TensorMap],
    ) -> tuple[list[System], dict[str, TensorMap], dict[str, TensorMap]]:
        """
        Transform function that merges blocks that only differ in atom types.

        :param systems: List of systems.
        :param targets: Dictionary containing the targets corresponding to the systems.
        :param extra: Dictionary containing any extra data.
        :return: The systems, updated targets and extra data.
        """
        new_targets = {**targets}
        new_extra = {**extra}

        for target_key, target_info in target_info_dict.items():
            if target_info.is_atomic_basis or getattr(target_info, "is_coupled_atomic_basis", False):
                new_targets[target_key] = merge_types(
                    targets[target_key], fill_value=fill_value
                )

        for extra_key, extra_info in extra_data_info_dict.items():
            if extra_info.is_atomic_basis or getattr(target_info, "is_coupled_atomic_basis", False):
                new_extra[extra_key] = merge_types(
                    extra[extra_key], fill_value=fill_value
                )

        return systems, new_targets, new_extra

    return transform
