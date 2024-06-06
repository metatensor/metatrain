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

        # Create the indices:
        self.indices = list(range(len(self)))

        # Shuffle the indices if requested
        if self.shuffle:
            np.random.shuffle(self.indices)

        self.reset()

    def reset(self):
        self.current_index = 0
        self.full_list = [batch for dl in self.dataloaders for batch in dl]

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self.indices):
            self.reset()  # Reset the index for the next iteration
            raise StopIteration

        idx = self.indices[self.current_index]
        self.current_index += 1
        return self.full_list[idx]

    def __len__(self):
        """Total number of batches in all dataloaders.

        This returns the total number of batches in all dataloaders
        (as opposed to the total number of samples or the number of
        individual dataloaders).

        :return: the total number of batches in all dataloaders
        """
        return sum(len(dl) for dl in self.dataloaders)
