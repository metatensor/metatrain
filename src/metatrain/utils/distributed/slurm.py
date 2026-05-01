import logging
import os

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
