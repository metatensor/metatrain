from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import metatensor.learn
import torch
from metatensor.torch import TensorMap
from torch import Generator, default_generator
from torch.utils.data import Subset, random_split


class Dataset:
    """A version of the `metatensor.learn.Dataset` class that allows for
    the use of `mtm::` prefixes in the keys of the dictionary. See
    https://github.com/lab-cosmo/metatensor/issues/621.

    It is important to note that, instead of named tuples, this class
    accepts and returns dictionaries.

    :param dict: A dictionary with the data to be stored in the dataset.
    """

    def __init__(self, dict: Dict):

        new_dict = {}
        for key, value in dict.items():
            key = key.replace("mtm::", "mtm_")
            new_dict[key] = value

        self.mts_learn_dataset = metatensor.learn.Dataset(**new_dict)

    def __getitem__(self, idx: int) -> Dict:

        mts_dataset_item = self.mts_learn_dataset[idx]._asdict()
        new_dict = {}
        for key, value in mts_dataset_item.items():
            key = key.replace("mtm_", "mtm::")
            new_dict[key] = value

        return new_dict

    def __len__(self) -> int:
        return len(self.mts_learn_dataset)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


@dataclass
class TargetInfo:
    """A class that contains information about a target.

    :param quantity: The quantity of the target.
    :param unit: The unit of the target.
    :param per_atom: Whether the target is a per-atom quantity.
    :param gradients: Gradients of the target that are defined
        in the current dataset.
    """

    quantity: str
    unit: str = ""
    per_atom: bool = False
    gradients: List[str] = field(default_factory=list)


@dataclass
class DatasetInfo:
    """A class that contains information about one or more datasets.

    This dataclass is used to communicate additional dataset details to the
    training functions of the individual models.

    :param length_unit: unit of length used in the dataset
    :param all_types: all possible atom in the dataset
    :param targets: information about targets in the dataset
    """

    length_unit: str
    all_types: List[int]
    targets: Dict[str, TargetInfo]


def get_all_types(datasets: Union[Dataset, List[Dataset]]) -> List[int]:
    """
    Returns the list of all types present in a dataset or list of datasets.

    :param datasets: the dataset, or list of datasets
    :returns: sorted list of all types present in the datasets
    """

    if not isinstance(datasets, list):
        datasets = [datasets]

    # Iterate over all single instances of the dataset:
    types = []
    for dataset in datasets:
        for index in range(len(dataset)):
            system = dataset[index]["system"]
            types += system.types.tolist()

    # Remove duplicates and sort:
    result = list(set(types))
    result.sort()

    return result


def get_all_targets(datasets: Union[Dataset, List[Dataset]]) -> List[str]:
    """
    Returns the list of all targets present in a dataset or list of datasets.

    :param datasets: the dataset(s).
    :returns: list of targets present in the dataset(s), sorted according
        to the ``sort()`` method of Python lists.
    """

    if not isinstance(datasets, list):
        datasets = [datasets]

    # The following does not work because the `dataset` can also
    # be a `Subset` object:
    # return list(dataset.targets.keys())

    # Iterate over all single instances of the dataset:
    target_names = []
    for dataset in datasets:
        for sample in dataset:
            sample.pop("system")  # system not needed
            target_names += list(sample.keys())

    # Remove duplicates:
    result = list(set(target_names))
    result.sort()

    return result


def collate_fn(batch: List[Dict[str, Any]]) -> Tuple[List, Dict[str, TensorMap]]:
    """
    Wraps `group_and_join` to
    return the data fields as a list of systems, and a dictionary of nameed
    targets.
    """

    collated_targets = group_and_join(batch)
    systems = collated_targets.pop("system")
    return systems, collated_targets


def check_datasets(train_datasets: List[Dataset], validation_datasets: List[Dataset]):
    """Check that the training and validation sets are compatible with one another

    Although these checks will not fit all use cases, most models would be expected
    to be able to use this function.

    :param train_datasets: A list of training datasets to check.
    :param validation_datasets: A list of validation datasets to check
    :raises TypeError: If the ``dtype`` within the datasets are inconsistent.
    :raises ValueError: If the `validation_datasets` has a target that is not present in
        the ``train_datasets``.
    :raises ValueError: If the training or validation set contains chemical species
        or targets that are not present in the training set
    """
    # Check that system `dtypes` are consistent within datasets
    desired_dtype = train_datasets[0][0]["system"].positions.dtype
    msg = f"`dtype` between datasets is inconsistent, found {desired_dtype} and "
    for train_dataset in train_datasets:
        actual_dtype = train_dataset[0]["system"].positions.dtype
        if actual_dtype != desired_dtype:
            raise TypeError(f"{msg}{actual_dtype} found in `train_datasets`")

    for validation_dataset in validation_datasets:
        actual_dtype = validation_dataset[0]["system"].positions.dtype
        if actual_dtype != desired_dtype:
            raise TypeError(f"{msg}{actual_dtype} found in `validation_datasets`")

    # Get all targets in the training and validation sets:
    train_targets = get_all_targets(train_datasets)
    validation_targets = get_all_targets(validation_datasets)

    # Check that the validation sets do not have targets that are not in the
    # training sets:
    for target in validation_targets:
        if target not in train_targets:
            raise ValueError(
                f"The validation dataset has a target ({target}) that is not present "
                "in the training dataset."
            )
    # Get all the species in the training and validation sets:
    all_training_species = get_all_types(train_datasets)
    all_validation_species = get_all_types(validation_datasets)

    # Check that the validation sets do not have species that are not in the
    # training sets:
    for species in all_validation_species:
        if species not in all_training_species:
            raise ValueError(
                f"The validation dataset has a species ({species}) that is not in the "
                "training dataset. This could be a result of a random train/validation "
                "split. You can avoid this by providing a validation dataset manually."
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


def group_and_join(
    batch: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Same as metatenor.learn.data.group_and_join, but joins dicts and not named tuples.

    :param batch: A list of dictionaries, each containing the data for a single sample.

    :returns: A single dictionary with the data fields joined together among all
        samples.
    """
    data: List[Union[TensorMap, torch.Tensor]] = []
    names = batch[0].keys()
    for name, f in zip(names, zip(*(item.values() for item in batch))):
        if name == "sample_id":  # special case, keep as is
            data.append(f)
            continue

        if isinstance(f[0], torch.ScriptObject) and f[0]._has_method(
            "keys_to_properties"
        ):  # inferred metatensor.torch.TensorMap type
            data.append(metatensor.torch.join(f, axis="samples"))
        elif isinstance(f[0], torch.Tensor):  # torch.Tensor type
            data.append(torch.vstack(f))
        else:  # otherwise just keep as a list
            data.append(f)

    return {name: value for name, value in zip(names, data)}
