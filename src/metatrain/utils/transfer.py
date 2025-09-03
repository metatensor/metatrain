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

    systems = [system.to(dtype=dtype, device=device, non_blocking=True) for system in systems]
    targets = {
        key: value.to(dtype=dtype, device=device, non_blocking=True) for key, value in targets.items()
    }
    if extra_data is not None:
        extra_data = {
            key: value.to(dtype=dtype, device=device, non_blocking=True)
            for key, value in extra_data.items()
        }

    return systems, targets, extra_data
