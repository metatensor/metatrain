from typing import Dict, List

import torch
from metatensor.torch import TensorMap
from metatomic.torch import System


@torch.jit.script
def tensormap_to_dtype(  # pragma: no cover
    tensormap_dict: Dict[str, TensorMap],
    dtype: torch.dtype,
):
    """
    Changes the data type of the TensorMaps to the specified floating point data type.

    :param tensormap_dict: Dictionary of TensorMaps.
    :param dtype: Desired floating point data type.
    """

    tensormap_dict = {
        key: value.to(dtype=dtype) for key, value in tensormap_dict.items()
    }

    return tensormap_dict


@torch.jit.script
def tensormap_to_device(  # pragma: no cover
    tensormap_dict: Dict[str, TensorMap],
    device: torch.device,
):
    """
    Moves the TensorMaps to the specified device.

    :param tensormap_dict: Dictionary of TensorMaps.
    :param device: Device to move to.
    """

    tensormap_dict = {
        key: value.to(device=device) for key, value in tensormap_dict.items()
    }

    return tensormap_dict


@torch.jit.script
def systems_and_targets_to_device(  # pragma: no cover
    systems: List[System],
    targets: Dict[str, TensorMap],
    device: torch.device,
):
    """
    Transfers the systems and targets to the specified device.

    :param systems: List of systems.
    :param targets: Dictionary of targets.
    :param device: Device to transfer to.
    """

    systems = [system.to(device=device) for system in systems]
    targets = tensormap_to_device(tensormap_dict=targets, device=device)
    return systems, targets


@torch.jit.script
def systems_and_targets_to_dtype(  # pragma: no cover
    systems: List[System],
    targets: Dict[str, TensorMap],
    dtype: torch.dtype,
):
    """
    Changes the systems and targets to the specified floating point data type.

    :param systems: List of systems.
    :param targets: Dictionary of targets.
    :param dtype: Desired floating point data type.
    """

    systems = [system.to(dtype=dtype) for system in systems]
    targets = tensormap_to_dtype(tensormap_dict=targets, dtype=dtype)
    return systems, targets
