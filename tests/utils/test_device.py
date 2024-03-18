import pytest
import torch

from metatensor.models.utils.devices import pick_devices


def test_pick_devices_cpu():
    available_devices = [torch.device("cpu")]
    architecture_devices = ["cpu"]
    assert pick_devices("cpu", available_devices, architecture_devices) == [
        torch.device("cpu")
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_pick_devices_gpu():
    available_devices = [torch.device("cpu"), torch.device("cuda:0")]
    architecture_devices = ["cuda", "cpu"]
    assert pick_devices("gpu", available_devices, architecture_devices) == [
        torch.device("cuda")
    ]


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="CUDA is not available or there is only one GPU",
)
def test_pick_devices_multi_gpu():
    available_devices = [
        torch.device("cpu"),
        torch.device("cuda:0"),
        torch.device("cuda:1"),
    ]
    architecture_devices = ["cpu", "cuda", "multi-cuda"]
    assert pick_devices("multi-gpu", available_devices, architecture_devices) == [
        torch.device("cuda:0"),
        torch.device("cuda:1"),
    ]


@pytest.mark.skipif(
    not (torch.backends.mps.is_built() and not torch.backends.mps.is_available()),
    reason="MPS is not available",
)
def test_pick_devices_mps():
    available_devices = [torch.device("cpu"), torch.device("mps")]
    architecture_devices = ["cpu", "mps"]
    assert pick_devices("mps", available_devices, architecture_devices) == [
        torch.device("mps")
    ]


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="CUDA is not available or there is only one GPU",
)
def test_pick_devices_multi_cuda():
    available_devices = [torch.device("cpu"), torch.device("cuda:0")]
    architecture_devices = ["cpu", "cuda", "multi-cuda"]
    assert pick_devices("multi-cuda", available_devices, architecture_devices) == [
        torch.device("cuda:0")
    ]


def test_pick_devices_cuda_no_cuda():
    available_devices = [torch.device("cpu")]
    architecture_devices = ["cpu"]
    with pytest.raises(ValueError, match="not available on this system"):
        pick_devices("cuda", available_devices, architecture_devices)


def test_pick_devices_multi_gpu_single_cuda():
    available_devices = [torch.device("cpu"), torch.device("cuda:0")]
    architecture_devices = ["cpu", "cuda"]
    with pytest.raises(ValueError, match="please use `gpu` or `cuda` instead"):
        pick_devices("multi-gpu", available_devices, architecture_devices)


def test_pick_devices_warning():
    available_devices = [torch.device("cpu"), torch.device("cuda:0")]
    architecture_devices = ["cuda", "cpu"]
    with pytest.warns(UserWarning, match="but the chosen architecture prefers"):
        pick_devices("cpu", available_devices, architecture_devices)


def test_pick_devices_invalid_device():
    available_devices = [torch.device("cpu")]
    architecture_devices = ["cpu"]
    with pytest.raises(ValueError, match="Unsupported device: `invalid`"):
        pick_devices("invalid", available_devices, architecture_devices)
