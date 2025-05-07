"""Test device selection.

Use pytest monkeypatching functions to perform some tests for GPU devices even though no
GPU might be present during the tests.

Some tests that require one or more GPUs to be present are located at the bottom of this
file.
"""

import pytest
import torch

from metatrain.utils.devices import pick_devices


def is_true() -> bool:
    return True


def is_false() -> bool:
    return False


@pytest.mark.parametrize("desired_device", ["cpu", None])
def test_pick_devices(desired_device):
    picked_devices = pick_devices(["cpu"], desired_device)
    assert picked_devices == [torch.device("cpu")]


@pytest.mark.parametrize("desired_device", ["cuda", None])
def test_pick_devices_cuda(desired_device, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", is_true)

    picked_devices = pick_devices(["cuda", "cpu"], desired_device)

    assert picked_devices == [torch.device("cuda")]


def test_pick_devices_prefer_architecture(monkeypatch):
    """Use architecture's preferred device if several matching devices are available."""
    monkeypatch.setattr(torch.cuda, "is_available", is_true)
    monkeypatch.setattr(torch.backends.mps, "is_built", is_true)
    monkeypatch.setattr(torch.backends.mps, "is_available", is_true)

    picked_devices = pick_devices(["cuda", "cpu"])

    assert picked_devices == [torch.device("cuda")]


@pytest.mark.parametrize("desired_device", ["mps", None])
def test_pick_devices_mps(desired_device, monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_built", is_true)
    monkeypatch.setattr(torch.backends.mps, "is_available", is_true)

    picked_devices = pick_devices(["mps", "cpu"], desired_device)

    assert picked_devices == [torch.device("mps")]


def test_no_matching_device(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_built", is_false)
    monkeypatch.setattr(torch.backends.mps, "is_available", is_false)

    match = (
        "No matching device found! The architecture requires cuda, mps; but your "
        "system only has cpu."
    )
    with pytest.raises(ValueError, match=match):
        pick_devices(["cuda", "mps"])


def test_pick_devices_unsupported_by_architecture(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", is_true)
    match = (
        "Desired device 'cuda' name resolved to "
        "'cuda' is not supported by the selected "
        "architecture. Please choose from cpu."
    )
    with pytest.raises(ValueError, match=match):
        pick_devices(["cpu"], "cuda")


@pytest.mark.parametrize("desired_device", ["multi-cuda", "multi-gpu"])
def test_pick_devices_multi_error(desired_device, monkeypatch):
    def device_count() -> int:
        return 1

    monkeypatch.setattr(torch.cuda, "is_available", is_true)
    monkeypatch.setattr(torch.cuda, "device_count", device_count)

    match = (
        f"Desired device '{desired_device}' name resolved to 'multi-cuda'"
        " is not supported by the selected your current system."
        " Please choose from cpu."
    )
    with pytest.raises(ValueError, match=match):
        pick_devices(["multi-cuda", "cpu"], desired_device=desired_device)


def test_pick_devices_preferred_warning(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_built", is_true)
    monkeypatch.setattr(torch.backends.mps, "is_available", is_true)

    match = (
        "Device 'cpu' - name resolved to 'cpu', requested,"
        " but 'mps' is preferred by the architecture"
        " and available on current system."
    )
    with pytest.warns(UserWarning, match=match):
        pick_devices(["mps", "cpu", "cuda"], desired_device="cpu")


def test_pick_devices_gpu_cuda_map(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", is_true)

    picked_devices = pick_devices(["cuda", "cpu"], "gpu")
    assert picked_devices == [torch.device("cuda")]


def test_pick_devices_no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", is_false)

    match = "Requested 'cuda' device, but cuda is not available."
    with pytest.raises(ValueError, match=match):
        pick_devices(["cuda", "cpu"], "cuda")


def test_pick_devices_gpu_mps_map(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_built", is_true)
    monkeypatch.setattr(torch.backends.mps, "is_available", is_true)

    picked_devices = pick_devices(["mps", "cpu"], "gpu")
    assert picked_devices == [torch.device("mps")]


@pytest.mark.parametrize(
    "is_built, is_available", [(is_true, is_false), (is_false, is_true)]
)
def test_pick_devices_no_mps(monkeypatch, is_built, is_available):
    monkeypatch.setattr(torch.backends.mps, "is_built", is_built)
    monkeypatch.setattr(torch.backends.mps, "is_available", is_available)

    match = "Requested 'mps' device, but mps is not available."
    with pytest.raises(ValueError, match=match):
        pick_devices(["mps", "cpu"], "mps")


@pytest.mark.parametrize("desired_device", ["multi-cuda", "multi-gpu"])
def test_pick_devices_multi_cuda(desired_device, monkeypatch):
    def device_count() -> int:
        return 2

    monkeypatch.setattr(torch.cuda, "is_available", is_true)
    monkeypatch.setattr(torch.cuda, "device_count", device_count)

    picked_devices = pick_devices(["multi-cuda", "cpu", "cuda"], desired_device)
    assert picked_devices == [
        torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
    ]


@pytest.mark.parametrize(
    "cuda_is_available, mps_is_build, mps_is_available",
    [
        (is_false, is_false, is_false),
        (is_false, is_true, is_false),
        (is_false, is_false, is_true),
    ],
)
def test_pick_devices_gpu_not_available(
    cuda_is_available, mps_is_build, mps_is_available, monkeypatch
):
    monkeypatch.setattr(torch.cuda, "is_available", cuda_is_available)
    monkeypatch.setattr(torch.backends.mps, "is_built", mps_is_build)
    monkeypatch.setattr(torch.backends.mps, "is_available", mps_is_available)

    with pytest.raises(ValueError, match="Requested 'gpu' device, but found no GPU"):
        pick_devices(["mps", "cpu"], "gpu")


def test_multi_gpu_warning(monkeypatch):
    def device_count() -> int:
        return 2

    monkeypatch.setattr(torch.cuda, "is_available", is_true)
    monkeypatch.setattr(torch.cuda, "device_count", device_count)

    match = (
        "Requested single 'cuda' device by specifying 'cuda' but current system "
        "has 2 cuda devices and architecture supports multi-gpu training. "
        "Consider using 'multi-gpu' to accelerate training."
    )
    with pytest.warns(UserWarning, match=match):
        picked_devices = pick_devices(["cuda", "multi-cuda", "cpu"], "cuda")

    assert picked_devices == [torch.device("cuda")]
