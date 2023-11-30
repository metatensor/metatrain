from typing import Dict, List

import metatensor.torch
import rascaline.torch
import torch
from metatensor.torch import Labels, TensorMap


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, structures: List[rascaline.torch.System], targets: Dict[str, TensorMap]
    ):
        """Creates a dataset from a list of `rascaline.torch.System`s and
        a list of dictionaries of `TensorMap`s."""

        for tensor_map in targets.values():
            n_structures = (
                torch.max(tensor_map.block(0).samples["structure"]).item() + 1
            )
            if n_structures != len(structures):
                raise ValueError(
                    f"Number of structures in input ({len(structures)}) and "
                    f"output ({n_structures}) must be the same"
                )

        self.structures = structures
        self.targets = targets

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.structures)

    def __getitem__(self, index):
        """
        Generates one sample of data.
        Args:
            index: The index of the item in the dataset.
        Returns:
            A tuple containing the structure and targets for the given index.
        """
        structure = self.structures[index]

        structure_index_samples = Labels(
            names=["structure"],
            values=torch.tensor([[index]]),  # must be a 2D-array
        )

        targets = {}
        for name, tensor_map in self.targets.items():
            targets[name] = metatensor.torch.slice(
                tensor_map, "samples", structure_index_samples
            )

        return structure, targets


def collate_fn(batch):
    """
    Creates a batch from a list of samples.
    Args:
        batch: A list of samples, where each sample is a tuple containing a
            structure and targets.
    Returns:
        A tuple containing the structures and targets for the batch.
    """

    structures = [sample[0] for sample in batch]
    targets = {}
    for name in batch[0][1].keys():
        targets[name] = metatensor.torch.join(
            [sample[1][name] for sample in batch], "samples"
        )

    return structures, targets
