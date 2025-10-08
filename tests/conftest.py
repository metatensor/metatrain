import math
import os


def pytest_xdist_auto_num_workers():
    n_processes = os.cpu_count() or 1
    return min(12, math.ceil(n_processes * 0.8))
