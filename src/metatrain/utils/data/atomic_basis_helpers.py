from typing import Callable, Dict, List, Optional, Tuple

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from .target_info import TargetInfo


# ===== General utilities


def reindex_to_batch_index(
    tensor: TensorMap,
    system_ids: torch.tensor,
) -> TensorMap:
    """
    Reindex the system ids in the samples of the blocks of the tensor to have batch ids.

    :param tensor: the input TensorMap with system ids in the samples to reindex.
    :param system_ids: Tensor containing the actual system ids for each sample in the
        input ``tensor``.

    :return: the output TensorMap with batch ids in the samples instead of system ids.
    """
    id_mapping = torch.ones(system_ids.max().item() + 1, dtype=int) * -1
    for new_id, old_id in enumerate(system_ids):
        id_mapping[old_id] = new_id

    blocks = []
    for block in tensor:
        system_ids = block.samples.values[:, 0]
        batch_id = id_mapping[system_ids]
        block = TensorBlock(
            samples=Labels(
                block.samples.names,
                torch.hstack(
                    [
                        batch_id.reshape(-1, 1),
                        block.samples.values[:, 1:],
                    ]
                ),
            ),
            components=block.components,
            properties=block.properties,
            values=block.values,
        )
        blocks.append(block)
    return TensorMap(tensor.keys, blocks)


def get_per_atom_sample_labels(
    systems: List[System],
) -> Labels:
    """
    Builds the atom sample labels for the input ``systems``, assuming the systems have a
    batch index (0, ..., n_batch - 1).

    :param systems: List of systems to build the per-atom sample labels for.
    :return: The per-atom sample labels.
    """
    system_indices = torch.concatenate(
        [
            torch.full(
                (len(system),),
                i_system,
                device=systems[0].device,
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
                        device=systems[0].device,
                    )
                    for system in systems
                ],
            ),
        ],
        dim=1,
    )
    sample_labels = Labels(
        names=["system", "atom"],
        values=sample_values,
    )
    return sample_labels


# ===== Densification utilities (keys with atom types to samples)


def _densify_per_atom_atomic_basis_target(
    tensor: TensorMap,
    layout: TensorMap,
    fill_value: float = torch.nan,
) -> TensorMap:
    """
    Densify the per-atom atomic basis target by moving the "atom_type" key dimension to
    the samples, creating a padded property dimension according to the maximum property
    size for each irrep.

    :param tensor: the per-atom atomic basis target TensorMap to densify.
    :param layout: the layout TensorMap defining the global basis set.
    :param fill_value: the value to use for filling in the padded values when
        densifying.

    :return: the densified per-atom atomic basis target TensorMap.
    """

    # First ensure that the tensor has all keys present in the layout tensor (i.e. the
    # global basis set definition). If any blocks aren't present, they are added as
    # zero-sample blocks with the correct components and properties.
    blocks = []
    for key, layout_block in layout.items():
        if key in tensor.keys:
            block = tensor.block(key)
        else:
            block = layout_block.copy()
            assert len(block.samples) == 0
        blocks.append(block)

    tensor = TensorMap(layout.keys, blocks)

    # Now densification can be done.

    # =====
    # TODO: the following is a manual densification, but this will be replaced with
    # `keys_to_samples(..., fill_value)` once publicly available in
    # metatensor-operations.
    # return mts.keys_to_samples(tensor, fill_value=fill_value, sort_samples=True)
    # =====

    # =====
    # For now, implement a manual densification:
    # =====

    # First, identify the "atom_type"-like and non-"atom_type"-like key dimensions.
    type_indices = [
        i for i, name in enumerate(tensor.keys.names) if name.endswith("atom_type")
    ]
    non_type_indices = [
        i for i, name in enumerate(tensor.keys.names) if not name.endswith("atom_type")
    ]
    type_names = [tensor.keys.names[i] for i in type_indices]

    # Using the layout TensorMap, build the union of the property labels values across
    # all atom types
    union_properties = {}
    for key, block in layout.items():
        key_vals = tuple([key.values[i].item() for i in non_type_indices])
        if key_vals not in union_properties:
            union_properties[key_vals] = block.properties
        else:
            union_properties[key_vals] = union_properties[key_vals].union(
                block.properties
            )

    # For each block, pad the properties using the dense properties
    padded_blocks = []
    for key, block in tensor.items():
        key_vals = tuple([key.values[i].item() for i in non_type_indices])
        properties = union_properties[key_vals]

        # Create a values array filled with the fill value
        padded_values = torch.full(
            (
                len(block.samples),
                *[len(c) for c in block.components],
                len(properties),
            ),
            fill_value,
            dtype=block.values.dtype,
        )

        # Now broadcast the existing values to the new shape
        properties_mask = properties.select(block.properties)
        padded_values[..., properties_mask] = block.values
        padded_block = TensorBlock(
            values=padded_values,
            samples=block.samples,
            components=block.components,
            properties=properties,
        )
        padded_blocks.append(padded_block)

    tensor = TensorMap(tensor.keys, padded_blocks)

    # Now move the "atom_type"-like key dimension to the samples and remove them
    tensor = tensor.keys_to_samples(type_names, sort_samples=True)
    for name in type_names:
        tensor = mts.remove_dimension(tensor, "samples", name)

    return tensor


def densify_atomic_basis_target(
    tensor: TensorMap,
    layout: TensorMap,
    fill_value: float = torch.nan,
) -> TensorMap:
    """
    Densify the atomic basis target by moving any "atom_type"-like key dimensions to the
    samples, creating a padded property dimension according to the maximum property size
    for each irrep.

    :param tensor: the atomic basis target TensorMap to densify.
    :param layout: the layout TensorMap defining the global basis set (i.e. the union of
        all blocks that should be present in the output).
    :param fill_value: the value to use for filling in the padded values when
        densifying.

    :return: the densified atomic basis target TensorMap.
    """
    if "atom" in tensor.sample_names:
        return _densify_per_atom_atomic_basis_target(tensor, layout, fill_value)

    raise NotImplementedError(
        "Currently only densification of per-atom atomic basis targets is implemented."
    )


def _pad_samples_per_atom_atomic_basis_target(
    systems: List[System],
    tensor: TensorMap,
) -> TensorMap:

    sample_labels = get_per_atom_sample_labels(systems)
    new_blocks = []
    for block in tensor:
        new_vals = torch.full(
            (len(sample_labels), *block.values.shape[1:]),
            fill_value=torch.nan,
            dtype=block.values.dtype,
        )
        sample_mask = sample_labels.select(block.samples)
        new_vals[sample_mask] = block.values
        new_block = TensorBlock(
            values=new_vals,
            samples=sample_labels,
            components=block.components,
            properties=block.properties,
        )
        new_blocks.append(new_block)

    return TensorMap(tensor.keys, new_blocks)


def pad_samples_atomic_basis_target(
    systems: List[System],
    tensor: TensorMap,
) -> TensorMap:
    """
    Pad the samples of the atomic basis target to have the same number of samples for
    each block.

    :param systems: List of systems in the batch
    :param tensor: the atomic basis target TensorMap to pad.

    :return: the padded atomic basis target TensorMap
    """

    if "atom" in tensor.sample_names:
        return _pad_samples_per_atom_atomic_basis_target(systems, tensor)
    raise NotImplementedError(
        "Currently only padding of per-atom atomic basis targets is implemented."
    )


def prepare_atomic_basis_targets(
    systems: List[System],
    system_ids: torch.Tensor,
    tensor: TensorMap,
    layout: TensorMap,
    fill_value: float = torch.nan,
) -> TensorMap:
    """
    Prepare the atomic basis targets for batching by reindexing to batch ids, densifying
    (moving "atom_type" key dimensions to the samples) and padding the samples.

    :param systems: List of systems in the batch.
    :param system_ids: Tensor containing the system ids for each sample in the input
        ``tensor``.
    :param tensor: the atomic basis target TensorMap to prepare.
    :param layout: the layout TensorMap defining the global basis set (i.e. the union of
        all blocks that should be present in the output).
    :param fill_value: the value to use for filling in the padded values when densifying
        and padding. Default is NaN, but can be set to 0 if desired (e.g. for
        classification targets).
    :return: the prepared atomic basis target TensorMap with batch ids, densified and
        padded.
    """

    # Reindex to batch ids
    tensor = reindex_to_batch_index(tensor, system_ids)

    # Densify: "atom type" key dimensions -> samples
    tensor = densify_atomic_basis_target(tensor, layout, fill_value)

    # Pad samples
    tensor = pad_samples_atomic_basis_target(systems, tensor)

    return tensor


# ===== Sparsification utilities (atom types back to keys)


def _sparsify_per_atom_atomic_basis_target(
    systems: List[System],
    tensor: TensorMap,
    layout: TensorMap,
    atom_types_batch: Optional[torch.Tensor] = None,
) -> TensorMap:
    """
    Sparsify the per-atom atomic basis target by creating blocks with an explicit
    "atom_type" dimension. The dense blocks of the input ``tensor`` are therefore sliced
    according to atom type of each atom in the samples.

    :param systems: List of systems in the batch.
    :param tensor: the atomic basis target TensorMap to sparsify.
    :param layout: the layout TensorMap defining the global basis set.
    :param atom_types_batch: Optional tensor containing the atom types for each sample
        in the batch. If not provided, these are inferred from the systems.
    :return: the sparsified atomic basis target TensorMap
    """
    if atom_types_batch is None:
        # Get the atom types for each sample in the batch from the systems
        atom_types_batch = torch.cat(
            [system.types for system in systems],
            dim=0,
        )

    # Sparsify by moving the "atom_type" from the samples to the keys
    unique_types = torch.unique(atom_types_batch)
    atom_type_masks: Dict[int, torch.Tensor] = {}
    for atom_type in unique_types:
        atom_type_masks[atom_type.item()] = atom_types_batch == atom_type

    new_keys: List[torch.Tensor] = []
    sparse_blocks: List[TensorBlock] = []
    for key, block in tensor.items():
        for atom_type in unique_types:
            new_key = torch.cat([key.values, atom_type.view(1)], dim=0)
            sparse_block = TensorBlock(
                values=block.values[atom_type_masks[atom_type.item()]],
                samples=Labels(
                    block.samples.names,
                    block.samples.values[atom_type_masks[atom_type.item()]],
                ),
                components=block.components,
                properties=block.properties,
            )

            new_keys.append(new_key)
            sparse_blocks.append(sparse_block)

    tensor = TensorMap(
        Labels(names=tensor.keys.names + ["atom_type"], values=torch.vstack(new_keys)),
        sparse_blocks,
    )

    # Now unpad the properties. Iterate over keys of the layout to automatically filter
    # out any blocks created that aren't in the basis set definition.
    key_vals: List[torch.Tensor] = []
    unpadded_blocks: List[TensorBlock] = []
    for key, layout_block in layout.items():
        if key not in tensor.keys:
            # can happen if this atom type isn't present in the batch
            assert not torch.any(unique_types == key["atom_type"])
            continue

        key_vals.append(key.values)
        block = tensor[key]

        layout_properties = layout_block.properties.to(block.properties.device)

        properties_mask = block.properties.select(layout_properties)
        # Do block.values[..., properties_mask] in a torchscriptable way.
        if block.values.ndim == 3:
            values = block.values[:, :, properties_mask]
        elif block.values.ndim == 4:
            values = block.values[:, :, :, properties_mask]
        else:
            raise ValueError(
                "Tensorblocks with more than 2 component dimensions can't be "
                "sparsified with the current implementation."
            )

        unpadded_block = TensorBlock(
            values=values,
            samples=block.samples,
            components=block.components,
            properties=layout_properties,
        )
        unpadded_blocks.append(unpadded_block)

    return TensorMap(
        Labels(
            layout.keys.names,
            torch.vstack(key_vals).to(unpadded_blocks[0].values.device),
        ),
        unpadded_blocks,
    )


def sparsify_atomic_basis_target(
    systems: List[System],
    tensor: TensorMap,
    layout: TensorMap,
    atom_types_batch: Optional[torch.Tensor] = None,
) -> TensorMap:
    """
    Sparsify the atomic basis target by creating blocks with an explicit "atom_type"
    dimension. The dense blocks of the input ``tensor`` are therefore sliced according
    to atom type of each atom in the samples.

    :param systems: List of systems in the batch.
    :param tensor: the atomic basis target TensorMap to sparsify.
    :param layout: the layout TensorMap defining the global basis set (i.e. the union of
        all blocks that should be present in the sparsified output).
    :param atom_types_batch: Optional tensor containing the atom types for each sample
        in the batch. If not provided, these are inferred from the systems.

    :return: the sparsified atomic basis target TensorMap
    """
    if "atom" in tensor.sample_names:
        return _sparsify_per_atom_atomic_basis_target(
            systems, tensor, layout, atom_types_batch
        )

    raise NotImplementedError(
        "Currently only sparsification of per-atom atomic basis targets is implemented."
    )


# ===== dataloader transforms


def get_reindex_to_batch_index_transform(
    target_info_dict: dict[str, TargetInfo],
    extra_data_info_dict: dict[str, TargetInfo],
) -> Callable:
    """
    Get a function that reindexes the systems to have batch ids.

    :param target_info_dict: Dictionary mapping target names to TargetInfo objects.
    :param extra_data_info_dict: Dictionary mapping extra data names to TargetInfo
        objects.

    :return: A function that takes in systems, targets and extra data, and returns the
        systems, targets and extra data with reindexed batch ids.
    """

    def transform(
        systems: List[System],
        targets: Dict[str, TensorMap],
        extra: Dict[str, TensorMap],
    ) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
        """
        Transform function that reindexes the systems to have batch ids, modifying
        in-place. Only applied to atomic basis targets and extra data.

        :param systems: List of systems.
        :param targets: Dictionary containing the targets corresponding to the systems.
        :param extra: Dictionary containing any extra data.
        :return: The systems, targets and extra data with reindexed system ids.
        """
        assert "mtt::aux::system_index" in extra
        for name, tensor in targets.items():
            if name in target_info_dict and target_info_dict[name].is_atomic_basis:
                targets[name] = reindex_to_batch_index(
                    tensor,
                    extra["mtt::aux::system_index"][0]
                    .values[:, 0]
                    .to(dtype=torch.int64),
                )

        for name, tensor in extra.items():
            if (
                name in extra_data_info_dict
                and extra_data_info_dict[name].is_atomic_basis
            ):
                extra[name] = reindex_to_batch_index(
                    tensor,
                    extra["mtt::aux::system_index"][0]
                    .values[:, 0]
                    .to(dtype=torch.int64),
                )

        return systems, targets, extra

    return transform


def get_prepare_atomic_basis_targets_transform(
    target_info_dict: dict[str, TargetInfo],
    extra_data_info_dict: dict[str, TargetInfo],
) -> Tuple[Callable, Callable]:
    """
    Get a function that prepares the atomic basis targets for batching by reindexing to
    batch ids, densifying and padding.

    :param target_info_dict: Dictionary mapping target names to TargetInfo objects.
    :param extra_data_info_dict: Dictionary mapping extra data names to TargetInfo
        objects.

    :return: A function that takes in systems, targets and extra data, and returns the
        systems, targets and extra data with prepared atomic basis targets.
    """

    def transform(
        systems: List[System],
        targets: Dict[str, TensorMap],
        extra: Dict[str, TensorMap],
    ) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
        """
        Transform function that prepares the atomic basis targets for batching by
        reindexing to batch ids, densifying and padding, modifying in-place.

        :param systems: List of systems.
        :param targets: Dictionary containing the targets corresponding to the systems.
        :param extra: Dictionary containing any extra data.
        :return: The systems, targets and extra data with prepared atomic basis targets.
        """
        for name, tensor in targets.items():
            if name in target_info_dict and target_info_dict[name].is_atomic_basis:
                assert "mtt::aux::system_index" in extra
                system_ids = (
                    extra["mtt::aux::system_index"][0]
                    .values[:, 0]
                    .to(dtype=torch.int64)
                )

                targets[name] = prepare_atomic_basis_targets(
                    systems,
                    system_ids,
                    tensor,
                    target_info_dict[name].layout,
                    fill_value=torch.nan,
                )

        for name, tensor in extra.items():
            if (
                name in extra_data_info_dict
                and extra_data_info_dict[name].is_atomic_basis
            ):
                assert "mtt::aux::system_index" in extra
                system_ids = (
                    extra["mtt::aux::system_index"][0]
                    .values[:, 0]
                    .to(dtype=torch.int64)
                )

                extra[name] = prepare_atomic_basis_targets(
                    systems,
                    system_ids,
                    tensor,
                    extra_data_info_dict[name].layout,
                    fill_value=torch.nan,
                )

        return systems, targets, extra

    def reverse_transform(
        systems: List[System],
        targets: Dict[str, TensorMap],
        extra: Dict[str, TensorMap],
    ) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
        """
        Reverse transform function that unpads, undensifies and reindexes the atomic
        basis targets, modifying in-place.

        :param systems: List of systems.
        :param targets: Dictionary containing the targets corresponding to the systems.
        :param extra: Dictionary containing any extra data.
        :return: The systems, targets and extra data with unprepared atomic basis
            targets.
        """
        for name, tensor in targets.items():
            if name in target_info_dict and target_info_dict[name].is_atomic_basis:
                targets[name] = sparsify_atomic_basis_target(
                    systems, tensor, target_info_dict[name].layout
                )

        for name, tensor in extra.items():
            if (
                name in extra_data_info_dict
                and extra_data_info_dict[name].is_atomic_basis
            ):
                extra[name] = sparsify_atomic_basis_target(
                    systems, tensor, extra_data_info_dict[name].layout
                )

        return systems, targets, extra

    return transform, reverse_transform
