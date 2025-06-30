from typing import Dict, List, Optional

import torch
from metatensor.torch import TensorMap
from metatomic.torch import System


@torch.jit.script
def systems_and_tensormap_dict_to_device(  # pragma: no cover
    systems: List[System],
    targets: Dict[str, TensorMap],
    device: torch.device,
    extra_data: Optional[Dict[str, TensorMap]] = None,
):
    """
    Transfers the systems and targets to the specified device.

    :param systems: List of systems.
    :param targets: Dictionary of targets.
    :param device: Device to transfer to.
    """

    systems = [system.to(device=device) for system in systems]
    targets = {key: value.to(device=device) for key, value in targets.items()}

    if extra_data is not None:
        extra_data = {key: value.to(device=device) for key, value in extra_data.items()}

    return systems, targets, extra_data


@torch.jit.script
def systems_and_tensormap_dict_to_dtype(  # pragma: no cover
    systems: List[System],
    targets: Dict[str, TensorMap],
    dtype: torch.dtype,
    extra_data: Optional[Dict[str, TensorMap]] = None,
):
    """
    Changes the systems and targets to the specified floating point data type.

    :param systems: List of systems.
    :param targets: Dictionary of targets.
    :param dtype: Desired floating point data type.
    """

    systems = [system.to(dtype=dtype) for system in systems]
    targets = {key: value.to(dtype=dtype) for key, value in targets.items()}

    if extra_data is not None:
        extra_data = {key: value.to(dtype=dtype) for key, value in extra_data.items()}

    return systems, targets, extra_data
