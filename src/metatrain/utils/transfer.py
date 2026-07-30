from typing import Any, Dict, List, Optional, Tuple

import torch
from metatensor.torch import TensorMap
from metatomic.torch import System

from metatrain.utils.data.raw_payload import RawExtraPayload, split_raw_payloads

from . import torch_jit_script_unless_coverage


@torch_jit_script_unless_coverage
def _batch_to_tensors(
    systems: List[System],
    targets: Dict[str, TensorMap],
    extra_data: Optional[Dict[str, TensorMap]] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tuple[List[System], Dict[str, TensorMap], Optional[Dict[str, TensorMap]]]:
    """
    Changes the systems and targets to the specified floating point data type.

    :param systems: List of systems.
    :param targets: Dictionary of targets.
    :param extra_data: Optional dictionary of extra data.
    :param dtype: Desired floating point data type.
    :param device: Device to move the data to.

    :return: The systems, targets, and extra data with moved to the specified device
        and with the desired data type.
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
            for (key, value), _dtype in zip(extra_data.items(), new_dtypes, strict=True)
        }

    return systems, targets, extra_data


def batch_to(
    systems: List[System],
    targets: Dict[str, TensorMap],
    extra_data: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tuple[List[System], Dict[str, TensorMap], Optional[Dict[str, Any]]]:
    """
    Move systems, targets and extra data to a device and dtype.

    A thin Python wrapper around the TorchScript-compiled ``_batch_to_tensors``.
    :class:`~metatrain.utils.data.raw_payload.RawExtraPayload` entries are arbitrary
    Python objects, which a ``ScriptFunction`` cannot accept, so they are split out,
    moved through their own ``to``, and merged back. Keeping the wrapper under the
    original name means no caller has to change.

    Raw payloads are moved to ``device`` but **not** cast to ``dtype``: they are not
    target values, and their precision is chosen where they are built.

    :param systems: List of systems.
    :param targets: Dictionary of targets.
    :param extra_data: Optional dictionary of extra data.
    :param dtype: Desired floating point data type.
    :param device: Device to move the data to.
    :return: The systems, targets and extra data, moved.
    """
    raw: Dict[str, RawExtraPayload] = {}
    if extra_data is not None:
        extra_data, raw = split_raw_payloads(extra_data)

    systems, targets, extra_data = _batch_to_tensors(
        systems, targets, extra_data, dtype, device
    )

    if raw:
        assert extra_data is not None
        for name, payload in raw.items():
            extra_data[name] = payload.to(device=device)
    return systems, targets, extra_data
