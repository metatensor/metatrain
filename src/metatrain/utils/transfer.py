from typing import Dict, List, Optional

import torch
from metatensor.torch import TensorMap
from metatomic.torch import System

from . import torch_jit_script_unless_coverage


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

    # non-blocking transfers can cause bugs in other cases
    non_blocking = (device.type == "cuda") if (device is not None) else False

    systems = [
        system.to(dtype=dtype, device=device, non_blocking=non_blocking)
        for system in systems
    ]
    targets = {
        key: value.to(dtype=dtype, device=device, non_blocking=non_blocking)
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
            key: value.to(dtype=_dtype, device=device, non_blocking=non_blocking)
            for (key, value), _dtype in zip(extra_data.items(), new_dtypes)
        }

    return systems, targets, extra_data
