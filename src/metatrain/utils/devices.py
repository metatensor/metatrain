import warnings
from typing import List, Optional

import torch


def _mps_is_available() -> bool:
    # require `torch.backends.mps.is_available()` for a reasonable check in torch<2.0
    return torch.backends.mps.is_built() and torch.backends.mps.is_available()


def pick_devices(
    architecture_devices: List[str],
    desired_device: Optional[str] = None,
) -> List[torch.device]:
    """Pick (best) devices for training.

    The choice is made on the intersection of the ``architecture_devices`` and the
    available devices on the current system. If no ``desired_device`` is provided the
    first device of this intersection will be returned.

    :param architecture_devices: Devices supported by the architecture. The list should
        be sorted by the preference of the architecture while the most preferred device
        should be first and the least one last.
    :param desired_device: desired device by the user. For example, ``"cpu"``,
        "``cuda``", ``"multi-gpu"``, etc.
    """

    available_devices = ["cpu"]
    if torch.cuda.is_available():
        available_devices.append("cuda")
        if torch.cuda.device_count() > 1:
            available_devices.append("multi-cuda")
    if _mps_is_available():
        available_devices.append("mps")

    # intersect between available and architecture's devices. keep order of architecture
    possible_devices = [d for d in architecture_devices if d in available_devices]

    if not possible_devices:
        raise ValueError(
            f"No matching device found! The architecture requires "
            f"{', '.join(architecture_devices)}; but your system only has "
            f"{', '.join(available_devices)}."
        )

    # If desired device given compare the possible devices and try to find a match
    if desired_device is None:
        desired_device = possible_devices[0]
    else:
        desired_device = desired_device.lower()

    # convert "gpu" and "multi-gpu" to "cuda" or "mps" if available
    if desired_device == "gpu":
        if torch.cuda.is_available():
            desired_device = "cuda"
        elif _mps_is_available():
            desired_device = "mps"
        else:
            raise ValueError(
                "Requested 'gpu' device, but found no GPU (CUDA or MPS) devices."
            )
    elif desired_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested 'cuda' device, but cuda is not available.")
    elif desired_device == "mps" and not _mps_is_available():
        raise ValueError("Requested 'mps' device, but mps is not available.")

    if desired_device == "multi-gpu":
        desired_device = "multi-cuda"

    if desired_device not in architecture_devices:
        raise ValueError(
            f"Desired device {desired_device!r} is not supported by the selected "
            f"architecture. Please choose from {', '.join(possible_devices)}."
        )

    if desired_device not in available_devices:
        raise ValueError(
            f"Desired device {desired_device!r} is not supported on "
            f"your current system. Please choose from {', '.join(possible_devices)}."
        )

    if possible_devices.index(desired_device) > 0:
        warnings.warn(
            f"Device {desired_device!r} requested, but {possible_devices[0]!r} is "
            "prefferred by the architecture and available on current system.",
            stacklevel=2,
        )

    if (
        desired_device == "cuda"
        and torch.cuda.device_count() > 1
        and any(d in possible_devices for d in ["multi-cuda", "multi_gpu"])
    ):
        warnings.warn(
            "Requested single 'cuda' device but current system has "
            f"{torch.cuda.device_count()} cuda devices and architecture supports "
            "multi-gpu training. Consider using 'multi-gpu' to accelerate "
            "training.",
            stacklevel=2,
        )

    # convert the requested device to a list of torch devices
    if desired_device == "multi-cuda":
        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    else:
        return [torch.device(desired_device)]
