import logging
from typing import List, Tuple

import torch
from metatensor.torch import TensorMap
from metatensor.torch.atomistic import System
from omegaconf import DictConfig

from ..target_info import TargetInfo, get_energy_target_info, get_generic_target_info


logger = logging.getLogger(__name__)


def read_metatensor_systems(filename: str) -> List[System]:
    """Read system information using metatensor.

    :param filename: name of the file to read

    :raises NotImplementedError: Serialization of systems is not yet
        available in metatensor.
    """
    raise NotImplementedError("Reading metatensor systems is not yet implemented.")


def _wrapped_metatensor_read(filename) -> List[TensorMap]:
    try:
        return torch.load(filename)
    except Exception as e:
        raise ValueError(f"Failed to read '{filename}' with torch: {e}") from e


def read_metatensor_energy(target: DictConfig) -> Tuple[List[TensorMap], TargetInfo]:
    tensor_maps = _wrapped_metatensor_read(target["read_from"])

    has_position_gradients = []
    has_strain_gradients = []
    for tensor_map in tensor_maps:
        if len(tensor_map) != 1:
            raise ValueError("Energy TensorMaps should have exactly one block.")
        has_position_gradients.append(
            "positions" in tensor_map.block().gradients_list()
        )
        has_strain_gradients.append("strain" in tensor_map.block().gradients_list())

    if not (all(has_position_gradients) or any(has_position_gradients)):
        raise ValueError(
            "Found a mix of targets with and without position gradients. "
            "Either all targets should have position gradients or none."
        )
    if not (all(has_strain_gradients) or any(has_strain_gradients)):
        raise ValueError(
            "Found a mix of targets with and without strain gradients. "
            "Either all targets should have strain gradients or none."
        )

    add_position_gradients = all(has_position_gradients)
    add_strain_gradients = all(has_strain_gradients)
    target_info = get_energy_target_info(
        target, add_position_gradients, add_strain_gradients
    )

    # now check all the expected metadata (from target_info.layout) matches
    # the actual metadata in the tensor maps
    _check_tensor_maps_metadata(tensor_maps, target_info.layout)

    return tensor_maps, target_info


def read_metatensor_generic(target: DictConfig) -> Tuple[List[TensorMap], TargetInfo]:
    tensor_maps = _wrapped_metatensor_read(target["read_from"])

    for tensor_map in tensor_maps:
        for block in tensor_map.blocks():
            if len(block.gradients_list()) > 0:
                raise ValueError("Only energy targets can have gradient blocks.")

    target_info = get_generic_target_info(target)

    _check_tensor_maps_metadata(tensor_maps, target_info.layout)

    return tensor_maps, target_info


def _check_tensor_maps_metadata(tensor_maps: List[TensorMap], layout: TensorMap):
    for i, tensor_map in enumerate(tensor_maps):
        if tensor_map.keys != layout.keys:
            raise ValueError(
                f"Unexpected keys in metatensor targets at index {i}: "
                f"expected: {layout.keys} "
                f"actual: {tensor_map.keys}"
            )
        for key in layout.keys:
            block = tensor_map.block(key)
            block_from_layout = tensor_map.block(key)
            if block.labels.names != block_from_layout.labels.names:
                raise ValueError(
                    f"Unexpected samples in metatensor targets at index {i}: "
                    f"expected: {block_from_layout.labels.names} "
                    f"actual: {block.labels.names}"
                )
            if block.components != block_from_layout.components:
                raise ValueError(
                    f"Unexpected components in metatensor targets at index {i}: "
                    f"expected: {block_from_layout.components} "
                    f"actual: {block.components}"
                )
            if block.properties != block_from_layout.properties:
                raise ValueError(
                    f"Unexpected properties in metatensor targets at index {i}: "
                    f"expected: {block_from_layout.properties} "
                    f"actual: {block.properties}"
                )
            if set(block.gradients_list()) != set(block_from_layout.gradients_list()):
                raise ValueError(
                    f"Unexpected gradients in metatensor targets at index {i}: "
                    f"expected: {block_from_layout.gradients_list()} "
                    f"actual: {block.gradients_list()}"
                )
            for name in block_from_layout.gradients_list():
                gradient_block = block.gradient(name)
                gradient_block_from_layout = block_from_layout.gradient(name)
                if (
                    gradient_block.labels.names
                    != gradient_block_from_layout.labels.names
                ):
                    raise ValueError(
                        f"Unexpected samples in metatensor targets at index {i} "
                        f"for `{name}` gradient block: "
                        f"expected: {gradient_block_from_layout.labels.names} "
                        f"actual: {gradient_block.labels.names}"
                    )
                if gradient_block.components != gradient_block_from_layout.components:
                    raise ValueError(
                        f"Unexpected components in metatensor targets at index {i} "
                        f"for `{name}` gradient block: "
                        f"expected: {gradient_block_from_layout.components} "
                        f"actual: {gradient_block.components}"
                    )
                if gradient_block.properties != gradient_block_from_layout.properties:
                    raise ValueError(
                        f"Unexpected properties in metatensor targets at index {i} "
                        f"for `{name}` gradient block: "
                        f"expected: {gradient_block_from_layout.properties} "
                        f"actual: {gradient_block.properties}"
                    )
