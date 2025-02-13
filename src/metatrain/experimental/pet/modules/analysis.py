import math

import numpy as np


def get_structural_batch_size(structures, atomic_batch_size):
    sizes = [len(structure.get_positions()) for structure in structures]
    average_size = np.mean(sizes)
    return math.ceil(atomic_batch_size / average_size)


def convert_atomic_throughput(train_structures, atomic_throughput):
    sizes = [len(structure.get_positions()) for structure in train_structures]
    total_size = np.sum(sizes)
    return math.ceil(atomic_throughput / total_size)


def adapt_hypers(hypers, train_structures):
    if "STRUCTURAL_BATCH_SIZE" not in hypers.__dict__.keys():
        hypers.STRUCTURAL_BATCH_SIZE = get_structural_batch_size(
            train_structures, hypers.ATOMIC_BATCH_SIZE
        )

    if "EPOCH_NUM" not in hypers.__dict__.keys():
        hypers.EPOCH_NUM = convert_atomic_throughput(
            train_structures, hypers.EPOCH_NUM_ATOMIC
        )

    if "SCHEDULER_STEP_SIZE" not in hypers.__dict__.keys():
        hypers.SCHEDULER_STEP_SIZE = convert_atomic_throughput(
            train_structures, hypers.SCHEDULER_STEP_SIZE_ATOMIC
        )

    if "EPOCHS_WARMUP" not in hypers.__dict__.keys():
        hypers.EPOCHS_WARMUP = convert_atomic_throughput(
            train_structures, hypers.EPOCHS_WARMUP_ATOMIC
        )
