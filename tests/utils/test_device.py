"""Test device selection.

Use pytest monkeypatching functions to perform some tests for GPU devices even though no
GPU might be present during the tests.

Some tests that require one or more GPUs to be present are located at the bottom of this
file.
"""

from typing import List

import pytest
import torch

from metatensor.models.utils import devices
from metatensor.models.utils.devices import pick_devices


@pytest.mark.parametrize("desired_device", ["cpu", None])
def test_pick_devices(desired_device):
    picked_devices = pick_devices(["cpu"], desired_device)
    assert picked_devices == [torch.device("cpu")]


@pytest.mark.parametrize("desired_device", ["cuda", None])
def test_pick_devices_cuda(desired_device, monkeypatch):
    def _get_available_devices() -> List[str]:
        return ["cuda", "cpu"]

    monkeypatch.setattr(devices, "_get_available_devices", _get_available_devices)

    picked_devices = pick_devices(["cuda", "cpu"], desired_device)

    assert picked_devices == [torch.device("cuda")]


@pytest.mark.parametrize("desired_device", ["mps", None])
def test_pick_devices_mps(desired_device, monkeypatch):
    def _get_available_devices() -> List[str]:
        return ["mps", "cpu"]

    monkeypatch.setattr(devices, "_get_available_devices", _get_available_devices)

    picked_devices = pick_devices(["mps", "cpu"], desired_device)

    assert picked_devices == [torch.device("mps")]


@pytest.mark.parametrize("desired_device", ["multi-cuda", None])
def test_pick_devices__multi_cuda(desired_device, monkeypatch):
    def _get_available_devices() -> List[str]:
        return ["cuda:0", "cuda:1", "cpu"]

    monkeypatch.setattr(devices, "_get_available_devices", _get_available_devices)

    picked_devices = pick_devices(["cuda", "cpu"], desired_device)

    assert picked_devices == [torch.device("cuda:0"), torch.device("cuda:1")]


def test_pick_devices_unsoprted():
    match = "Unsupported desired device 'cuda'. Please choose from cpu."
    with pytest.raises(ValueError, match=match):
        pick_devices(["cpu"], "cuda")


def test_pick_devices_preferred_warning(monkeypatch):
    def _get_available_devices() -> List[str]:
        return ["mps", "cpu"]

    monkeypatch.setattr(devices, "_get_available_devices", _get_available_devices)

    match = "Device 'cpu' requested, but 'mps' is prefferred"
    with pytest.warns(UserWarning, match=match):
        pick_devices(["mps", "cpu", "cuda"], desired_device="cpu")


@pytest.mark.parametrize("desired_device", ["multi-cuda", "multi-gpu"])
def test_pick_devices_multi_error(desired_device, monkeypatch):
    def _get_available_devices() -> List[str]:
        return ["multi-cuda", "cuda", "cpu"]

    monkeypatch.setattr(devices, "_get_available_devices", _get_available_devices)

    with pytest.raises(ValueError, match="Requested device 'multi-gpu'"):
        pick_devices(["multi-cuda", "cpu"], desired_device=desired_device)


# Below tests that require specific devices to be present
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_pick_devices_gpu_cuda_map():
    picked_devices = pick_devices(["cuda", "cpu"], "gpu")
    assert picked_devices == [torch.device("cuda")]


@pytest.mark.skipif(
    not (torch.backends.mps.is_built() and torch.backends.mps.is_available()),
    reason="MPS is not available",
)
def test_pick_devices_gpu_mps_map():
    picked_devices = pick_devices(["mps", "cpu"], "gpu")
    assert picked_devices == [torch.device("mps")]


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="less than 2 CUDA devices")
@pytest.mark.parametrize("desired_device", ["multi-cuda", "multi-gpu"])
def test_pick_devices_multi_cuda(desired_device):
    picked_devices = pick_devices(["cpu", "cuda", "multi-cuda"], desired_device)
    assert picked_devices == [
        torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
    ]


@pytest.mark.skipif(
    torch.cuda.is_available()
    or (torch.backends.mps.is_built() and torch.backends.mps.is_available()),
    reason="GPU device available",
)
def test_pick_devices_gpu_not_available():
    with pytest.raises(ValueError, match="Requested 'gpu' device, but found no GPU"):
        pick_devices(["cuda", "cpu"], "gpu")
