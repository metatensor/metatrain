import pytest
import torch

from metatrain.utils.architectures import check_architecture_options, get_default_hypers
from metatrain.utils.distributed.slurm import (
    initialize_slurm_nccl_process_group,
    resolve_distributed,
)


def _set_slurm_env(monkeypatch, num_tasks):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_NTASKS", str(num_tasks))


def _pet_options(**training):
    options = get_default_hypers("pet", base_precision=32)
    options["training"].update(training)
    return options


def test_resolve_distributed_auto(monkeypatch):
    """The default "auto" enables distributed training exactly when the job
    runs under more than one SLURM task."""
    _set_slurm_env(monkeypatch, 16)
    assert resolve_distributed("auto") is True

    _set_slurm_env(monkeypatch, 1)
    assert resolve_distributed("auto") is False

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    assert resolve_distributed("auto") is False


def test_multitask_slurm_auto(monkeypatch):
    _set_slurm_env(monkeypatch, 16)
    check_architecture_options(name="pet", options=_pet_options())


def test_explicit_distributed_deprecated(monkeypatch):
    _set_slurm_env(monkeypatch, 1)
    with pytest.warns(DeprecationWarning, match="DEPRECATED"):
        check_architecture_options(name="pet", options=_pet_options(distributed=True))


def test_multitask_slurm_distributed_disabled(monkeypatch):
    """Multiple SLURM tasks with distributed training explicitly disabled must
    fail early: every task would otherwise run its own full copy of the
    training, all writing to the same output files."""
    _set_slurm_env(monkeypatch, 16)
    with (
        pytest.warns(DeprecationWarning, match="DEPRECATED"),
        pytest.raises(ValueError, match="distributed training is disabled"),
    ):
        check_architecture_options(name="pet", options=_pet_options(distributed=False))


def test_multitask_slurm_unsupported_architecture(monkeypatch):
    _set_slurm_env(monkeypatch, 16)
    with pytest.raises(ValueError, match="does not support distributed training"):
        check_architecture_options(
            name="composition", options=get_default_hypers("composition")
        )


def test_initialize_slurm_nccl_process_group_single_visible_gpu(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node[01-02]")
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setenv("SLURM_PROCID", "3")
    monkeypatch.setenv("SLURM_LOCALID", "3")

    monkeypatch.setattr(
        "metatrain.utils.distributed.slurm.hostlist.expand_hostlist",
        lambda nodelist: ["node01", "node02"],
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    calls = []

    def fake_set_device(device):
        calls.append(("set_device", device))

    def fake_init_process_group(*, backend, device_id):
        calls.append(("init_process_group", backend, device_id))

    monkeypatch.setattr(torch.cuda, "set_device", fake_set_device)
    monkeypatch.setattr(
        torch.distributed, "init_process_group", fake_init_process_group
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)

    device, world_size, rank = initialize_slurm_nccl_process_group(39591)

    assert device == torch.device("cuda", 0)
    assert world_size == 4
    assert rank == 3
    assert calls == [
        ("set_device", torch.device("cuda", 0)),
        ("init_process_group", "nccl", torch.device("cuda", 0)),
    ]


def test_initialize_slurm_nccl_process_group_multiple_visible_gpus(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "456")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node[03-04]")
    monkeypatch.setenv("SLURM_NTASKS", "8")
    monkeypatch.setenv("SLURM_PROCID", "2")
    monkeypatch.setenv("SLURM_LOCALID", "2")

    monkeypatch.setattr(
        "metatrain.utils.distributed.slurm.hostlist.expand_hostlist",
        lambda nodelist: ["node03", "node04"],
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    set_device_calls = []
    init_calls = []

    def fake_set_device(device):
        set_device_calls.append(device)

    def fake_init_process_group(*, backend, device_id):
        init_calls.append((backend, device_id))

    monkeypatch.setattr(torch.cuda, "set_device", fake_set_device)
    monkeypatch.setattr(
        torch.distributed, "init_process_group", fake_init_process_group
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 8)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 2)

    device, world_size, rank = initialize_slurm_nccl_process_group(39591)

    assert device == torch.device("cuda", 2)
    assert world_size == 8
    assert rank == 2
    assert set_device_calls == [torch.device("cuda", 2)]
    assert init_calls == [("nccl", torch.device("cuda", 2))]
