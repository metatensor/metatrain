from typing import Dict, List

import torch
from metatensor.torch import TensorMap
from metatensor.torch.atomistic import System


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
    targets = {key: value.to(device=device) for key, value in targets.items()}
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
    targets = {key: value.to(dtype=dtype) for key, value in targets.items()}
    return systems, targets
