import logging
import os
from typing import Dict, List, NamedTuple, Optional, Tuple

import metatensor.torch
import torch
from metatensor.learn.data import Dataset, group_and_join
from metatensor.learn.data.dataset import _BaseDataset
from metatensor.torch import TensorMap
from metatensor.torch.atomistic import ModelCapabilities
from torch import Generator, default_generator
from torch.utils.data import Subset, random_split


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


def get_all_species(datasets: List[_BaseDataset]) -> List[int]:
    """
    Returns the list of all species present in a list of datasets.

    Args:
        datasets: The datasets.

    Returns:
        The list of species present in the datasets.
    """

    # The following does not work because the `dataset` can also
    # be a `Subset` object:
    # species = []
    # for structure in dataset.structures:
    #     species += structure.species.tolist()
    # return list(set(species))

    # Iterate over all single instances of the dataset:
    species = []
    for dataset in datasets:
        for index in range(len(dataset)):
            structure = dataset[index][0]  # extract the structure from the NamedTuple
            species += structure.species.tolist()

    # Remove duplicates and sort:
    result = list(set(species))
    result.sort()

    return result


def get_all_targets(dataset: _BaseDataset) -> List[str]:
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
        sample = dataset[index]._asdict()  # NamedTuple -> dict
        sample.pop("structure")  # structure not needed
        target_names += list(sample.keys())

    # Remove duplicates:
    return list(set(target_names))


def collate_fn(batch: List[NamedTuple]) -> Tuple[List, Dict[str, TensorMap]]:
    """
    Wraps the `metatensor-learn` default collate function `group_and_join` to
    return the data fields as a list of structures, and a dictionary of nameed
    targets.
    """

    collated_targets = group_and_join(batch)._asdict()
    structures = collated_targets.pop("structure")
    return structures, collated_targets


def check_datasets(
    train_datasets: List[_BaseDataset],
    validation_datasets: List[_BaseDataset],
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
    all_training_species = get_all_species(train_datasets)
    all_validation_species = get_all_species(validation_datasets)

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


def _train_test_random_split(
    train_dataset: Dataset,
    train_size: float,
    test_size: float,
    generator: Optional[Generator] = default_generator,
) -> List[Subset]:
    if train_size <= 0:
        raise ValueError("Fraction of the train set is smaller or equal to 0!")

    # normalize fractions
    lengths = torch.tensor([train_size, test_size])
    lengths /= lengths.sum()

    return random_split(dataset=train_dataset, lengths=lengths, generator=generator)
