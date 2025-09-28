from typing import Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from . import torch_jit_script_unless_coverage


def _pin_memory_labels(labels: Labels) -> Labels:
    return Labels(names=labels.names, values=labels.values.pin_memory())


def _pin_memory_system(system: System) -> System:
    new_system = System(
        positions=system.positions.pin_memory(),
        types=system.types.pin_memory(),
        cell=system.cell.pin_memory(),
        pbc=system.pbc.pin_memory(),
    )
    for nl_options in system.known_neighbor_lists():
        nl = system.get_neighbor_list(nl_options)
        new_system.add_neighbor_list(
            nl_options,
            _pin_memory_tensorblock(nl),
        )
    for key in system.known_data():
        data = system.get_data(key)
        new_system.add_data(key, _pin_memory_tensormap(data))
    return new_system


def _pin_memory_tensorblock(tensor_block: TensorBlock) -> TensorBlock:
    new_tensor_block = TensorBlock(
        values=tensor_block.values.pin_memory(),
        samples=_pin_memory_labels(tensor_block.samples),
        components=[
            _pin_memory_labels(component) for component in tensor_block.components
        ],
        properties=_pin_memory_labels(tensor_block.properties),
    )
    # torchscript doesn't support recursive calls, we have to limit this to one level
    # of gradients, which is the only one we use in metatrain
    for gradient_name, gradient_block in tensor_block.gradients():
        new_tensor_block.add_gradient(
            gradient_name,
            TensorBlock(
                values=gradient_block.values.pin_memory(),
                samples=_pin_memory_labels(gradient_block.samples),
                components=[
                    _pin_memory_labels(component)
                    for component in gradient_block.components
                ],
                properties=_pin_memory_labels(gradient_block.properties),
            ),
        )
    return new_tensor_block


def _pin_memory_tensormap(tensor_map: TensorMap) -> TensorMap:
    new_keys = _pin_memory_labels(tensor_map.keys)
    new_blocks = [_pin_memory_tensorblock(block) for block in tensor_map.blocks()]
    return TensorMap(keys=new_keys, blocks=new_blocks)


def _pin_memory_batch(
    systems: List[System],
    targets: Dict[str, TensorMap],
    extra_data: Optional[Dict[str, TensorMap]],
):
    """Pin all memory in a batch to make transfer to GPU faster."""
    new_systems = [_pin_memory_system(system) for system in systems]
    new_targets = {k: _pin_memory_tensormap(v) for k, v in targets.items()}
    if extra_data is None:
        new_extra_data = None
    else:
        new_extra_data = {k: _pin_memory_tensormap(v) for k, v in extra_data.items()}
    return new_systems, new_targets, new_extra_data


@torch_jit_script_unless_coverage
def batch_to(
    systems: List[System],
    targets: Dict[str, TensorMap],
    extra_data: Optional[Dict[str, TensorMap]] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
):
    """
    Changes the systems and targets to the specified floating point data type.

    :param systems: List of systems.
    :param targets: Dictionary of targets.
    :param dtype: Desired floating point data type.
    """

    if device is not None:
        if device.type == "cuda":
            systems, targets, extra_data = _pin_memory_batch(
                systems, targets, extra_data
            )

    systems = [
        system.to(dtype=dtype, device=device, non_blocking=True) for system in systems
    ]
    targets = {
        key: value.to(dtype=dtype, device=device, non_blocking=True)
        for key, value in targets.items()
    }
    if extra_data is not None:
        new_dtypes: List[Optional[int]] = []
        for key in extra_data.keys():
            if key.endswith("_mask"):  # masks should always be boolean
                new_dtypes.append(torch.bool)
            else:
                new_dtypes.append(dtype)
        extra_data = {
            key: value.to(dtype=_dtype, device=device, non_blocking=True)
            for (key, value), _dtype in zip(extra_data.items(), new_dtypes)
        }

    return systems, targets, extra_data
