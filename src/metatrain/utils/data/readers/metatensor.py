from typing import List, Tuple

import metatensor.torch
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import System
from omegaconf import DictConfig

from ..target_info import TargetInfo, get_energy_target_info, get_generic_target_info


def read_systems(filename: str) -> List[System]:
    """Read system information using metatensor.

    :param filename: name of the file to read

    :raises NotImplementedError: Serialization of systems is not yet
        available in metatensor.
    """
    raise NotImplementedError("Reading metatensor systems is not yet implemented.")


def _wrapped_metatensor_read(filename) -> TensorMap:
    try:
        return metatensor.torch.load(filename)
    except Exception as e:
        raise ValueError(f"Failed to read '{filename}' with torch: {e}") from e


def read_energy(target: DictConfig) -> Tuple[TensorMap, TargetInfo]:
    tensor_map = _wrapped_metatensor_read(target["read_from"])

    if len(tensor_map) != 1:
        raise ValueError("Energy TensorMaps should have exactly one block.")

    add_position_gradients = target["forces"]
    add_strain_gradients = target["stress"] or target["virial"]
    target_info = get_energy_target_info(
        target, add_position_gradients, add_strain_gradients
    )

    # now check all the expected metadata (from target_info.layout) matches
    # the actual metadata in the tensor maps
    _check_tensor_map_metadata(tensor_map, target_info.layout)

    selections = [
        Labels(
            names=["system"],
            values=torch.tensor([[int(i)]]),
        )
        for i in torch.unique(
            torch.concatenate(
                [block.samples.column("system") for block in tensor_map.blocks()]
            )
        )
    ]
    tensor_maps = metatensor.torch.split(tensor_map, "samples", selections)
    return tensor_maps, target_info


def read_generic(target: DictConfig) -> Tuple[List[TensorMap], TargetInfo]:
    tensor_map = _wrapped_metatensor_read(target["read_from"])

    for block in tensor_map.blocks():
        if len(block.gradients_list()) > 0:
            raise ValueError("Only energy targets can have gradient blocks.")

    target_info = get_generic_target_info(target)
    _check_tensor_map_metadata(tensor_map, target_info.layout)

    # make sure that the properties of the target_info.layout also match the
    # actual properties of the tensor maps
    target_info.layout = _empty_tensor_map_like(tensor_map)

    selections = [
        Labels(
            names=["system"],
            values=torch.tensor([[int(i)]]),
        )
        for i in torch.unique(tensor_map.block(0).samples.column("system"))
    ]
    tensor_maps = metatensor.torch.split(tensor_map, "samples", selections)
    return tensor_maps, target_info


def _check_tensor_map_metadata(tensor_map: TensorMap, layout: TensorMap):
    if tensor_map.keys != layout.keys:
        raise ValueError(
            f"Unexpected keys in metatensor targets: "
            f"expected: {layout.keys} "
            f"actual: {tensor_map.keys}"
        )
    for key in layout.keys:
        block = tensor_map.block(key)
        block_from_layout = layout.block(key)
        if block.samples.names != block_from_layout.samples.names:
            raise ValueError(
                f"Unexpected samples in metatensor targets: "
                f"expected: {block_from_layout.samples.names} "
                f"actual: {block.samples.names}"
            )
        if block.components != block_from_layout.components:
            raise ValueError(
                f"Unexpected components in metatensor targets: "
                f"expected: {block_from_layout.components} "
                f"actual: {block.components}"
            )
        # the properties can be different from those of the default `TensorMap`
        # given by `get_generic_target_info`, so we don't check them
        if set(block.gradients_list()) != set(block_from_layout.gradients_list()):
            raise ValueError(
                f"Unexpected gradients in metatensor targets: "
                f"expected: {block_from_layout.gradients_list()} "
                f"actual: {block.gradients_list()}"
            )
        for name in block_from_layout.gradients_list():
            gradient_block = block.gradient(name)
            gradient_block_from_layout = block_from_layout.gradient(name)
            if gradient_block.labels.names != gradient_block_from_layout.labels.names:
                raise ValueError(
                    f"Unexpected samples in metatensor targets "
                    f"for `{name}` gradient block: "
                    f"expected: {gradient_block_from_layout.labels.names} "
                    f"actual: {gradient_block.labels.names}"
                )
            if gradient_block.components != gradient_block_from_layout.components:
                raise ValueError(
                    f"Unexpected components in metatensor targets "
                    f"for `{name}` gradient block: "
                    f"expected: {gradient_block_from_layout.components} "
                    f"actual: {gradient_block.components}"
                )


def _empty_tensor_map_like(tensor_map: TensorMap) -> TensorMap:
    new_keys = tensor_map.keys
    new_blocks: List[TensorBlock] = []
    for block in tensor_map.blocks():
        new_block = _empty_tensor_block_like(block)
        new_blocks.append(new_block)
    return TensorMap(keys=new_keys, blocks=new_blocks)


def _empty_tensor_block_like(tensor_block: TensorBlock) -> TensorBlock:
    new_block = TensorBlock(
        values=torch.empty(
            (0,) + tensor_block.values.shape[1:],
            dtype=torch.float64,  # metatensor can't serialize otherwise
            device=tensor_block.values.device,
        ),
        samples=Labels(
            names=tensor_block.samples.names,
            values=torch.empty(
                (0, tensor_block.samples.values.shape[1]),
                dtype=tensor_block.samples.values.dtype,
                device=tensor_block.samples.values.device,
            ),
        ),
        components=tensor_block.components,
        properties=tensor_block.properties,
    )
    for gradient_name, gradient in tensor_block.gradients():
        new_block.add_gradient(gradient_name, _empty_tensor_block_like(gradient))
    return new_block
