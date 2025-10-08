from .slurm import is_slurm, is_slurm_main_process


def is_main_process() -> bool:
    """
    Check if the current process is the main process.

    :return: True if the current process is the main process, False otherwise.
    """
    if is_slurm():
        return is_slurm_main_process()
    else:
        return True
