import os

import hostlist


def is_slurm():
    return ("SLURM_JOB_ID" in os.environ) and ("SLURM_PROCID" in os.environ)


def is_slurm_main_process():
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

    def __init__(self, port: int):
        self._setup_distr_env(port)
        self.master_addr = os.environ["MASTER_ADDR"]
        self.master_port = os.environ["MASTER_PORT"]
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])

    def _setup_distr_env(self, port: int):
        hostnames = hostlist.expand_hostlist(os.environ["SLURM_JOB_NODELIST"])
        os.environ["MASTER_ADDR"] = hostnames[0]  # set first node as master
        os.environ["MASTER_PORT"] = str(port)  # set port for communication
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
