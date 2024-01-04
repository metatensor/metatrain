import itertools

import numpy as np
import torch


class CombinedIterableDataset(torch.utils.data.IterableDataset):
    """
    Combines multiple dataloaders into a single iterable dataset.

    This is useful for combining multiple datasets into a single dataloader
    and learning from all of them simultaneously.
    """

    def __init__(self, dataloaders, shuffle):
        self.dataloaders = dataloaders
        self.shuffle = shuffle
        self.indices = self._create_indices()

    def _create_indices(self):
        # Create a list of (dataloader_idx, idx) tuples
        # for all indices in all dataloaders
        indices = [
            (i, dl_idx)
            for dl_idx, dl in enumerate(self.dataloaders)
            for i in range(len(dl))
        ]

        # Shuffle the indices if requested
        if self.shuffle:
            np.random.shuffle(indices)
        return indices

    def __iter__(self):
        for idx, dataloader_idx in self.indices:
            yield next(itertools.islice(self.dataloaders[dataloader_idx], idx, None))

    def __len__(self):
        return len(self.indices)


def combine_dataloaders(*dataloaders, shuffle=True):
    combined_dataset = CombinedIterableDataset(dataloaders, shuffle)
    return torch.utils.data.DataLoader(combined_dataset, batch_size=None)
