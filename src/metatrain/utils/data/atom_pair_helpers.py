from typing import Callable

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from metatrain.utils.data.target_info import TargetInfo


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
                    "Expected the input TensorMap to only contain one direction of the edges."
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
        :return: The systems, targets and extra data with bidirectional data for the edges.
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
