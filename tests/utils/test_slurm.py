import torch

from metatrain.utils.distributed.slurm import initialize_slurm_nccl_process_group


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
