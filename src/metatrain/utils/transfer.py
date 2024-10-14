from typing import Dict, List

import torch
from metatensor.torch import TensorMap
from metatensor.torch.atomistic import System


@torch.jit.script
def systems_and_targets_to_dtype_and_device(
    systems: List[System],
    targets: Dict[str, TensorMap],
    dtype: torch.dtype,
    device: torch.device,
):
    """
    Transfers the systems and targets to the specified dtype and device.

    :param systems: List of systems.
    :param targets: Dictionary of targets.
    :param dtype: Desired data type.
    :param device: Device to transfer to.
    """

    systems = [system.to(dtype=dtype, device=device) for system in systems]
    targets = {
        key: value.to(dtype=dtype, device=device) for key, value in targets.items()
    }
    return systems, targets
