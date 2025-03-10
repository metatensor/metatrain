from .slurm import is_slurm
import os


def is_distributed():
    if is_slurm():
        return int(os.environ["WORLD_SIZE"]) > 1
    else:
        return False
