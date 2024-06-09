from .slurm import is_slurm, is_slurm_main_process


def is_main_process():
    if is_slurm():
        return is_slurm_main_process()
    else:
        return True
