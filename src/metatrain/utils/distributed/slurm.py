import logging
import os
import warnings
from typing import Union

import hostlist
import torch
import torch.distributed


def is_slurm() -> bool:
    """
    Check if the code is running within a SLURM job.

    :return: True if running in a SLURM job, False otherwise.
    """
    return ("SLURM_JOB_ID" in os.environ) and ("SLURM_PROCID" in os.environ)


def is_slurm_main_process() -> bool:
    """
    Check if the current process is the main process in a SLURM job.

    :return: True if the current process is the main process, False otherwise.
    """
    return os.environ["SLURM_PROCID"] == "0"


def resolve_distributed(distributed: Union[bool, str]) -> bool:
    """
    Resolve the ``distributed`` hyperparameter to a boolean.

    The default value ``"auto"`` enables distributed training when running
    inside a SLURM job with more than one task. Explicit booleans override the
    detection, but are deprecated.

    :param distributed: The raw value of the ``distributed`` hyperparameter.
    :return: Whether to use distributed training.
    """
    if distributed == "auto":
        return is_slurm() and int(os.environ.get("SLURM_NTASKS", "1")) > 1
    warnings.warn(
        "DEPRECATED[distributed]: Setting the `distributed` option explicitly "
        "is deprecated and will be removed at some point. The default value "
        "'auto' enables distributed training automatically when running under "
        "more than one SLURM task.",
        DeprecationWarning,
        stacklevel=2,
    )
    return bool(distributed)


def check_slurm_distributed_config(
    architecture_name: str, training_hypers: dict
) -> None:
    """
    Check that a multi-task SLURM launch matches the distributed configuration.

    When ``mtt train`` is launched with more than one SLURM task while
    distributed training is disabled (or not supported by the architecture),
    every task silently runs its own full copy of the training, all writing to
    the same output files. Fail early with a clear message instead.

    :param architecture_name: Name of the architecture being trained.
    :param training_hypers: The architecture's training hyperparameters.
    :raises ValueError: If the job runs under more than one SLURM task while
        distributed training is disabled or unsupported.
    """
    if not is_slurm():
        return
    num_tasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if num_tasks <= 1:
        return
    if "distributed" not in training_hypers:
        raise ValueError(
            f"This job was launched with {num_tasks} SLURM tasks, but the "
            f"'{architecture_name}' architecture does not support distributed "
            "training: every task would run its own full copy of the same "
            "training. Please launch with a single task."
        )
    if training_hypers["distributed"] is False:
        raise ValueError(
            f"This job was launched with {num_tasks} SLURM tasks, but "
            "distributed training is disabled: every task would run its own "
            "full copy of the same training. Remove 'distributed: false' from "
            "the 'training' section of the architecture options (the default "
            "'auto' enables distributed training in multi-task SLURM jobs), "
            "or launch with a single task."
        )


class DistributedEnvironment:
    """
    Distributed environment for Slurm.

    This class sets up the distributed environment on Slurm. It reads
    the necessary environment variables and sets them for use in the
    PyTorch distributed utilities. Modified from
    https://github.com/Lumi-supercomputer/lumi-reframe-tests/blob/main/checks/apps/deeplearning/pytorch/src/pt_distr_env.py.

    :param port: The port to use for communication in the distributed
        environment.
    """  # noqa: E501, E262

    def __init__(self, port: int) -> None:
        self._setup_distr_env(port)
        self.master_addr = os.environ["MASTER_ADDR"]
        self.master_port = os.environ["MASTER_PORT"]
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])

    def _setup_distr_env(self, port: int) -> None:
        hostnames = hostlist.expand_hostlist(os.environ["SLURM_JOB_NODELIST"])
        os.environ["MASTER_ADDR"] = hostnames[0]  # set first node as master
        os.environ["MASTER_PORT"] = str(port)  # set port for communication
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]

        logging.info(
            f"Distributed environment set up with "
            f"MASTER_ADDR={os.environ['MASTER_ADDR']}, "
            f"MASTER_PORT={os.environ['MASTER_PORT']}, "
            f"WORLD_SIZE={os.environ['WORLD_SIZE']}, "
            f"RANK={os.environ['RANK']}, LOCAL_RANK={os.environ['LOCAL_RANK']}"
        )


def initialize_slurm_nccl_process_group(port: int) -> tuple[torch.device, int, int]:
    """
    Initialize the default NCCL process group for a Slurm-launched run.

    The device mapping follows the current metatrain convention: use the local rank
    modulo the number of visible CUDA devices so the setup works both when ranks see
    all GPUs on the node and when each rank only sees a single GPU.

    :param port: The port to use for communication in the distributed environment.
    :return: The local CUDA device, world size, and global rank.
    """

    distr_env = DistributedEnvironment(port)
    device_number = distr_env.local_rank % torch.cuda.device_count()
    device = torch.device("cuda", device_number)
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    return device, world_size, rank
