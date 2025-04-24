from typing import List

import numpy as np
import torch


class CombinedDataLoader:
    """
    Combines multiple dataloaders into a single dataloader.

    This is useful for learning from multiple datasets at the same time,
    each of which may have different batch sizes, properties, etc.

    :param dataloaders: list of dataloaders to combine
    :param shuffle: whether to shuffle the combined dataloader (this does not
        act on the individual batches, but it shuffles the order in which
        they are returned)

    :return: the combined dataloader
    """

    def __init__(self, dataloaders: List[torch.utils.data.DataLoader], shuffle: bool):
        self.dataloaders = dataloaders
        self.shuffle = shuffle

        # Create the indices. These contain which dataloader each batch comes from.
        # These will be shuffled later.
        self.indices = []
        for i, dl in enumerate(dataloaders):
            self.indices.extend([i] * len(dl))

        self.reset()

    def reset(self):
        self.dataloader_iterators = [iter(dl) for dl in self.dataloaders]
        self.current_index = 0
        # Shuffle the indices, if requested, for every new epoch
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self.indices):
            self.reset()  # Reset the index for the next iteration
            raise StopIteration

        idx = self.indices[self.current_index]
        self.current_index += 1
        return next(self.dataloader_iterators[idx])

    def __len__(self):
        """Total number of batches in all dataloaders.

        This returns the total number of batches in all dataloaders
        (as opposed to the total number of samples or the number of
        individual dataloaders).

        :return: the total number of batches in all dataloaders
        """
        return sum(len(dl) for dl in self.dataloaders)
