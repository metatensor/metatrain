import numpy as np


def dataloader(dataset, batch_size, shuffle=True):
    dataset_size = len(dataset)

    indices = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, dataset_size, batch_size):
        end_idx = min(start_idx + batch_size, dataset_size)
        batch_indices = indices[start_idx:end_idx]
        yield [dataset[i] for i in batch_indices]
