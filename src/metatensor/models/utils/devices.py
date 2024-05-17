import warnings
from typing import List, Optional

import torch


def _get_available_devices() -> List[str]:
    available_devices = ["cpu"]
    if torch.cuda.is_available():
        available_devices.append("cuda")
        if torch.cuda.device_count() > 1:
            available_devices.append("multi-cuda")
    # for torch<2.0 `torch.backends.mps.is_available()` is required for a reasonable
    # check.
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        available_devices.append("mps")

    return available_devices


def pick_devices(
    architecture_devices: List[str],
    desired_device: Optional[str] = None,
) -> List[torch.device]:
    """Pick (best) devices for training.

    The choice is made on the intersection of the ``architecture_devices`` and the
    available devices on the current system. If no ``desired_device`` is provided the
    first device of this intersection will be returned.

    :param architecture_devices: Devices supported by the architecture. The list should
        be sorted by the preference of the architecture while the most prefferred device
        should be first and the least one last.
    :param desired_device: desired device by the user
    """

    available_devices = _get_available_devices()

    # intersect between available and architecture's devices. keep order of architecture
    possible_devices = [d for d in architecture_devices if d in available_devices]

    # cpu device should always be available
    assert "cpu" in possible_devices

    # If desired device given compare the possible devices and try to find a match
    if desired_device is None:
        desired_device = possible_devices[0]
    else:
        desired_device = desired_device.lower()

        # convert "gpu" and "multi-gpu" to "cuda" or "mps" if available
        if desired_device == "gpu":
            if torch.cuda.is_available():
                desired_device = "cuda"
            elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
                desired_device = "mps"
            else:
                raise ValueError(
                    "Requested 'gpu' device, but found no GPU (CUDA or MPS) devices."
                )
        if desired_device == "multi-gpu":
            desired_device = "multi-cuda"

        if desired_device not in possible_devices:
            raise ValueError(
                f"Unsupported desired device {desired_device!r}. "
                f"Please choose from {', '.join(possible_devices)}."
            )
        if desired_device == "multi-cuda" and torch.cuda.device_count() < 2:
            raise ValueError(
                "Requested device 'multi-gpu' or 'multi-cuda', but found only one CUDA "
                "device. If you want to run on a single GPU, please use 'gpu' or "
                "'cuda' instead."
            )

        if possible_devices.index(desired_device) > 0:
            warnings.warn(
                f"Device {desired_device!r} requested, but {possible_devices[0]!r} is "
                "prefferred by the architecture and available on current system.",
                stacklevel=2,
            )

    # convert the requested device to a list of torch devices
    if desired_device == "multi-cuda":
        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    else:
        return [torch.device(desired_device)]
