import warnings
from typing import List

import torch


def get_available_devices() -> List[torch.device]:
    """Returns a list of available torch devices.

    This function returns a list of available torch devices, which can
    be used to specify the devices on which to run a model.

    :return: The list of available torch devices.
    """
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            devices.append(torch.device(f"cuda:{i}"))
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    return devices


def pick_devices(
    requested_device: str,
    available_devices: List[torch.device],
    architecture_devices: List[str],
) -> List[torch.device]:
    """Picks the devices to use for training.

    This function picks the devices to use for training based on the
    requested device, the available devices, and the list of devices
    supported by the architecture.

    The choice is based on the following logic. First, the requested
    device is checked to see if it is supported (i.e., one of "cpu",
    "cuda", "mps", "gpu", "multi-gpu", or "multi-cuda"). Then, the
    requested device is checked to see if it is available on the system.
    Finally, the requested device is checked to see if it is supported
    by the architecture. If the requested device is not supported by the
    architecture, a ValueError is raised. If the requested device is
    supported by the architecture, but a different device is preferred
    by the architecture and present on the system, a warning is issued.

    :param requested_device: The requested device.
    :param available_devices: The available devices.
    :param architecture_devices: The devices supported by the architecture.
    """

    requested_device = requested_device.lower()

    # first, we check that the requested device is supported
    if requested_device not in ["cpu", "cuda", "multi-cuda", "mps", "gpu", "multi-gpu"]:
        raise ValueError(
            f"Unsupported device: {requested_device}, please choose from "
            "cpu, cuda, mps, gpu, multi-gpu, multi-cuda"
        )

    # we convert "gpu" and "multi-gpu" to "cuda" or "mps" if available
    if requested_device == "gpu":
        if torch.cuda.is_available():
            requested_device = "cuda"
        elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
            requested_device = "mps"
        else:
            raise ValueError(
                "Requested `gpu` device, but found no GPU (CUDA or MPS) devices"
            )

    # we convert "multi-gpu" to "multi-cuda"
    if requested_device == "multi-gpu":
        requested_device = "multi-cuda"

    # check that the requested device is available
    available_device_types = [device.type for device in available_devices]
    available_device_strings = ["cpu"]  # always available
    if "cuda" in available_device_types:
        available_device_strings.append("cuda")
    if "mps" in available_device_types:
        available_device_strings.append("mps")
    if available_device_strings.count("cuda") > 1:
        available_device_strings.append("multi-cuda")

    if requested_device not in available_device_strings:
        if requested_device == "multi-cuda":
            if available_device_strings.count("cuda") == 0:
                raise ValueError(
                    "Requested device `multi-gpu` or `multi-cuda`, "
                    "but found no cuda devices"
                )
            else:
                raise ValueError(
                    "Requested device `multi-gpu` or `multi-cuda`, "
                    "but found only one cuda device. If you want to run on a "
                    "single GPU, please use `gpu` or `cuda` instead."
                )
        else:
            raise ValueError(
                f"Requested device {requested_device} is not available on this system"
            )

    # if the requested device is available, check it against the architecture's devices
    if requested_device not in architecture_devices:
        raise ValueError(
            f"The requested device `{requested_device}` is not supported by the chosen "
            f"architecture. Supported devices are {architecture_devices}."
        )

    # we check all the devices that come before the requested one in the
    # list of architecture devices. If any of them are available, we warn

    requested_device_index = architecture_devices.index(requested_device)
    for device in architecture_devices[:requested_device_index]:
        if device in available_device_strings:
            warnings.warn(
                f"Device `{requested_device}` was requested, but the chosen"
                f"architecture prefers `{device}`, which was also found on your "
                f"system. Consider using the `{device}` device.",
                stacklevel=2,
            )

    # finally, we convert the requested device to a list of devices
    if requested_device == "multi-cuda":
        return [device for device in available_devices if device.type == "cuda"]
    else:
        return [torch.device(requested_device)]
