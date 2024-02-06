import logging
import os
from typing import Dict, List, Tuple

import metatensor.torch
import numpy as np
import torch
from metatensor.torch import Labels, TensorMap
from metatensor.torch.atomistic import ModelCapabilities, System


logger = logging.getLogger(__name__)


if os.environ.get("METATENSOR_IMPORT_FOR_SPHINX", "0") == "1":
    # This is necessary to make the Sphinx documentation build
    def compiled_slice(a, b):
        pass

    def compiled_join(a, axis, remove_tensor_name):
        pass

else:
    compiled_slice = torch.jit.script(metatensor.torch.slice)
    compiled_join = torch.jit.script(metatensor.torch.join)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, structures: List[System], targets: Dict[str, TensorMap]):
        """
        Creates a dataset from a list of `metatensor.torch.atomistic.System`
        objects and a dictionary of targets where the keys are strings and
        the values are `TensorMap` objects.
        """

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

        sample_labels = Labels(
            names=["structure"],
            values=torch.tensor([index]).reshape(1, 1),
        )

        targets = {}
        for name, tensor_map in self.targets.items():
            targets[name] = compiled_slice(tensor_map, "samples", sample_labels)

        return structure, targets


def get_all_species(dataset: Dataset) -> List[int]:
    """
    Returns the list of all species present in the dataset.

    Args:
        dataset: The dataset.

    Returns:
        The list of species present in the dataset.
    """

    # The following does not work because the `dataset` can also
    # be a `Subset` object:
    # species = []
    # for structure in dataset.structures:
    #     species += structure.species.tolist()
    # return list(set(species))

    # Iterate over all single instances of the dataset:
    species = []
    for index in range(len(dataset)):
        structure, _ = dataset[index]
        species += structure.species.tolist()

    # Remove duplicates and sort:
    result = list(set(species))
    result.sort()

    return result


def get_all_targets(dataset: Dataset) -> List[str]:
    """
    Returns the list of all targets present in the dataset.

    Args:
        dataset: The dataset.

    Returns:
        The list of targets present in the dataset.
    """

    # The following does not work because the `dataset` can also
    # be a `Subset` object:
    # return list(dataset.targets.keys())

    # Iterate over all single instances of the dataset:
    target_names = []
    for index in range(len(dataset)):
        _, targets = dataset[index]
        target_names += list(targets.keys())

    # Remove duplicates:
    return list(set(target_names))


def collate_fn(batch: List[Tuple[System, Dict[str, TensorMap]]]):
    """
    Creates a batch from a list of samples.

    Args:
        batch: A list of samples, where each sample is a tuple containing a
            structure and targets.

    Returns:
        A tuple containing the structures and targets for the batch.
    """

    structures: List[System] = [sample[0] for sample in batch]
    # `join` will reorder the samples based on their structure number.
    # Let's reorder the list of structures in the same way:
    structure_samples = [
        list(sample[1].values())[0].block().samples.values.item() for sample in batch
    ]
    sorting_order = np.argsort(structure_samples)
    structures = [structures[index] for index in sorting_order]
    # TODO: use metatensor.learn for datasets/dataloaders, making sure the same
    # issues are handled correctly

    targets: Dict[str, TensorMap] = {}
    names = list(batch[0][1].keys())
    for name in names:
        targets[name] = compiled_join(
            [sample[1][name] for sample in batch],
            axis="samples",
            remove_tensor_name=True,
        )
    return structures, targets


def check_datasets(
    train_datasets: List[Dataset],
    validation_datasets: List[Dataset],
    capabilities: ModelCapabilities,
):
    """
    This is a helper function that checks that the training and validation sets
    are compatible with one another and with the model's capabilities. Although
    these checks will not fit all use cases, they will fit most.

    :param train_datasets: A list of training datasets.
    :param validation_datasets: A list of validation datasets.
    :param capabilities: The model's capabilities.

    :raises ValueError: If the training and validation sets are not compatible
        with the model's capabilities.
    """

    # Get all targets in the training and validation sets:
    train_targets = []
    for dataset in train_datasets:
        train_targets += get_all_targets(dataset)
    validation_targets = []
    for dataset in validation_datasets:
        validation_targets += get_all_targets(dataset)

    # Check that they are compatible with the model's capabilities:
    for target in train_targets + validation_targets:
        if target not in capabilities.outputs.keys():
            raise ValueError(f"The target {target} is not in the model's capabilities.")

    # Check that the validation sets do not have targets that are not in the
    # training sets:
    for target in validation_targets:
        if target not in train_targets:
            logger.warn(
                f"The validation dataset has a target ({target}) "
                "that is not in the training dataset."
            )

    # Get all the species in the training and validation sets:
    all_training_species = []
    for dataset in train_datasets:
        all_training_species += get_all_species(dataset)
    all_validation_species = []
    for dataset in validation_datasets:
        all_validation_species += get_all_species(dataset)

    # Check that they are compatible with the model's capabilities:
    for species in all_training_species + all_validation_species:
        if species not in capabilities.species:
            raise ValueError(
                f"The species {species} is not in the model's capabilities."
            )

    # Check that the validation sets do not have species that are not in the
    # training sets:
    for species in all_validation_species:
        if species not in all_training_species:
            logger.warn(
                f"The validation dataset has a species ({species}) "
                "that is not in the training dataset. This could be "
                "a result of a random train/validation split. You can "
                "avoid this by providing a validation dataset manually."
            )
