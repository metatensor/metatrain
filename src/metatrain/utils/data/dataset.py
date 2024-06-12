import itertools
import math
import warnings
from collections import UserDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import metatensor.learn
import torch
from metatensor.torch import TensorMap
from torch import Generator, default_generator

from ..external_naming import to_external_name


@dataclass
class TargetInfo:
    """A class that contains information about a target.

    :param quantity: The quantity of the target.
    :param unit: The unit of the target. If :py:obj:`None` the ``unit`` will be set to
        an empty string ``""``.
    :param per_atom: Whether the target is a per-atom quantity.
    :param gradients: Set of gradients of the target that are defined in the current
        dataset. Examples are ``"positions"`` or ``"strain"``.
    """

    quantity: str
    unit: str = ""
    per_atom: bool = False
    gradients: Set[str] = field(default_factory=set)

    def __post_init__(self):
        if self.unit is None:
            self.unit = ""

        # For compatibility with list convert to set
        self.gradients = set(self.gradients)

    def copy(self) -> "TargetInfo":
        """Return a shallow copy of the TargetInfo."""
        return TargetInfo(
            quantity=self.quantity,
            unit=self.unit,
            per_atom=self.per_atom,
            gradients=self.gradients.copy(),
        )

    def update(self, other: "TargetInfo") -> None:
        """Update this instance with the union of itself and ``other``.

        :raises ValueError: If ``quantity``, ``unit`` or ``per_atom`` do not match.
        """
        if self.quantity != other.quantity:
            raise ValueError(
                f"Can't update TargetInfo with a different `quantity`: "
                f"({self.quantity} != {other.quantity})"
            )

        if self.unit != other.unit:
            raise ValueError(
                f"Can't update TargetInfo with a different `unit`: "
                f"({self.unit} != {other.unit})"
            )

        if self.per_atom != other.per_atom:
            raise ValueError(
                f"Can't update TargetInfo with a different `per_atom` property: "
                f"({self.per_atom} != {other.per_atom})"
            )

        self.gradients = self.gradients.union(other.gradients)

    def union(self, other: "TargetInfo") -> "TargetInfo":
        """Return the union of this instance with ``other``."""
        new = self.copy()
        new.update(other)
        return new


class TargetInfoDict(UserDict):
    """
    A custom dictionary class for storing and managing ``TargetInfo`` instances.

    The subclass handles the update of :py:class:`TargetInfo` if a ``key`` is already
    present.
    """

    # We use a `UserDict` with special methods because a normal dict does not support
    # the update of nested instances.
    def __setitem__(self, key, value):
        if not isinstance(value, TargetInfo):
            raise ValueError("value to set is not a `TargetInfo` instance")
        if key in self:
            self[key].update(value)
        else:
            super().__setitem__(key, value)

    def __and__(self, other: "TargetInfoDict") -> "TargetInfoDict":
        return self.intersection(other)

    def __sub__(self, other: "TargetInfoDict") -> "TargetInfoDict":
        return self.difference(other)

    def union(self, other: "TargetInfoDict") -> "TargetInfoDict":
        """Union of this instance with ``other``."""
        new = self.copy()
        new.update(other)
        return new

    def intersection(self, other: "TargetInfoDict") -> "TargetInfoDict":
        """Intersection of the the two instances as a new ``TargetInfoDict``.

        (i.e. all elements that are in both sets.)

        :raises ValueError: If intersected items with the same key are not the same.
        """
        new_keys = self.keys() & other.keys()

        self_intersect = TargetInfoDict(**{key: self[key] for key in new_keys})
        other_intersect = TargetInfoDict(**{key: other[key] for key in new_keys})

        if self_intersect == other_intersect:
            return self_intersect
        else:
            raise ValueError(
                "Intersected items with the same key are not the same. Intersected "
                f"keys are {','.join(new_keys)}"
            )

    def difference(self, other: "TargetInfoDict") -> "TargetInfoDict":
        """Difference of two instances as a new ``TargetInfoDict``.

        (i.e. all elements that are in this set but not in the other.)
        """

        new_keys = self.keys() - other.keys()
        return TargetInfoDict(**{key: self[key] for key in new_keys})


@dataclass
class DatasetInfo:
    """A class that contains information about datasets.

    This dataclass is used to communicate additional dataset details to the
    training functions of the individual models.

    :param length_unit: Unit of length used in the dataset. If :py:obj:`None` the
        ``length_unit`` will be set to an empty string ``""``.
    :param atomic_types: Unordered set of all atomic types present in the dataset.

        .. note::

            ``atomic_types`` is a :py:class:`set` and **not ordered**. Use
            :py:func:`sorted` for an ordered :py:class:`list`.
    :param targets: Information about targets in the dataset.
    """

    length_unit: Union[None, str]
    atomic_types: Set[int]
    targets: TargetInfoDict

    def __post_init__(self):
        if self.length_unit is None:
            self.length_unit = ""

        # For compatibility with list convert to set
        self.atomic_types = set(self.atomic_types)

    def copy(self) -> "DatasetInfo":
        """Return a shallow copy of the DatasetInfo."""
        return DatasetInfo(
            length_unit=self.length_unit,
            atomic_types=self.atomic_types.copy(),
            targets=self.targets.copy(),
        )

    def update(self, other: "DatasetInfo") -> None:
        """Update this instance with the union of itself and ``other``.

        :raises ValueError: If the ``length_units`` are different.
        """
        if self.length_unit != other.length_unit:
            raise ValueError(
                "Can't update DatasetInfo with a different `length_unit`: "
                f"({self.length_unit} != {other.length_unit})"
            )

        self.atomic_types = self.atomic_types.union(other.atomic_types)
        self.targets = self.targets.union(other.targets)

    def union(self, other: "DatasetInfo") -> "DatasetInfo":
        """Return the union of this instance with ``other``."""
        new = self.copy()
        new.update(other)
        return new


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

    def get_stats(self, dataset_info: DatasetInfo) -> str:
        if hasattr(self, "_cached_stats"):
            return self._cached_stats  # type: ignore
        stats = _get_dataset_stats(self, dataset_info)
        self._cached_stats = stats
        return stats


class Subset(torch.utils.data.Subset):
    """
    A version of `torch.utils.data.Subset` containing a `get_stats` method
    allowing us to print information about atomistic datasets.
    """

    def get_stats(self, dataset_info: DatasetInfo) -> str:
        if hasattr(self, "_cached_stats"):
            return self._cached_stats  # type: ignore
        stats = _get_dataset_stats(self, dataset_info)
        self._cached_stats = stats
        return stats


def _get_dataset_stats(dataset: Union[Dataset, Subset], dataset_info: DatasetInfo):
    """Returns the statistics of a dataset or subset as a string."""

    dataset_len = len(dataset)
    stats = f"Dataset of size {dataset_len}"
    if dataset_len == 0:
        return stats

    target_names = []
    for key, tensor_map in dataset[0].items():
        if key == "system":
            continue
        target_names.append(key)
        gradients_list = tensor_map.block(0).gradients_list()
        for gradient in gradients_list:
            target_names.append(f"{key}_{gradient}_gradients")

    sums = {key: 0.0 for key in target_names}
    n_elements = {key: 0 for key in target_names}
    for sample in dataset:
        for key in target_names:
            if "_gradients" not in key:  # not a gradient
                tensors = [block.values for block in sample[key].blocks()]
            else:
                original_key = key.split("_")[0]
                gradient_name = key.replace(f"{original_key}_", "").replace(
                    "_gradients", ""
                )
                tensors = [
                    block.gradient(gradient_name).values
                    for block in sample[original_key].blocks()
                ]
            sums[key] += sum(tensor.sum() for tensor in tensors)
            n_elements[key] += sum(tensor.numel() for tensor in tensors)
    means = {key: sums[key] / n_elements[key] for key in target_names}

    sum_of_squared_residuals = {key: 0.0 for key in target_names}
    for sample in dataset:
        for key in target_names:
            if "_gradients" not in key:  # not a gradient
                tensors = [block.values for block in sample[key].blocks()]
            else:
                original_key = key.split("_")[0]
                gradient_name = key.replace(f"{original_key}_", "").replace(
                    "_gradients", ""
                )
                tensors = [
                    block.gradient(gradient_name).values
                    for block in sample[original_key].blocks()
                ]
            sum_of_squared_residuals[key] += sum(
                ((tensor - means[key]) ** 2).sum() for tensor in tensors
            )
    stds = {
        key: (sum_of_squared_residuals[key] / n_elements[key]) ** 0.5
        for key in target_names
    }

    stats += "\n    Mean and standard deviation of targets:"
    for key in target_names:
        stats += (
            f"\n    - {to_external_name(key, dataset_info.targets)}: "  # type: ignore
            f"mean={means[key]:.3e}, std={stds[key]:.3e}"
        )

    return stats


def get_atomic_types(datasets: Union[Dataset, List[Dataset]]) -> Set[int]:
    """List of all atomic types present in a dataset or list of datasets.

    :param datasets: the dataset, or list of datasets
    :returns: sorted list of all atomic types present in the datasets
    """

    if not isinstance(datasets, list):
        datasets = [datasets]

    types = []
    for dataset in datasets:
        for index in range(len(dataset)):
            system = dataset[index]["system"]
            types += system.types.tolist()

    return set(types)


def get_all_targets(datasets: Union[Dataset, List[Dataset]]) -> List[str]:
    """Sorted list of all unique targets present in a dataset or list of datasets.

    :param datasets: the dataset(s).
    :returns: Sorted list of all targets present in the dataset(s).
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

    return sorted(set(target_names))


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
    all_training_species = get_atomic_types(train_datasets)
    all_validation_species = get_atomic_types(validation_datasets)

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
) -> List[Dataset]:
    if train_size <= 0:
        raise ValueError("Fraction of the train set is smaller or equal to 0!")

    # normalize fractions
    lengths = torch.tensor([train_size, test_size])
    lengths /= lengths.sum()

    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(train_dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(train_dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. "
                    f"This might result in an empty dataset.",
                    UserWarning,
                    stacklevel=2,
                )

    # Cannot verify that train_dataset is Sized
    if sum(lengths) != len(train_dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = torch.randperm(sum(lengths), generator=generator).tolist()
    return [
        Subset(train_dataset, indices[offset - length : offset])
        for offset, length in zip(itertools.accumulate(lengths), lengths)
    ]


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
