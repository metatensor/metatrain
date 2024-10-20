import math
import warnings
from collections import UserDict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from metatensor.learn.data import Dataset, group_and_join
from metatensor.torch import TensorMap
from torch.utils.data import Subset

from ..external_naming import to_external_name
from ..units import get_gradient_units


class TargetInfo:
    """A class that contains information about a target.

    :param quantity: The quantity of the target.
    :param unit: The unit of the target. If :py:obj:`None` the ``unit`` will be set to
        an empty string ``""``.
    :param per_atom: Whether the target is a per-atom quantity.
    :param gradients: List containing the gradient names of the target that are present
        in the target. Examples are ``"positions"`` or ``"strain"``. ``gradients`` will
        be stored as a sorted list of **unique** gradients.
    """

    def __init__(
        self,
        quantity: str,
        unit: Union[None, str] = "",
        per_atom: bool = False,
        gradients: Optional[List[str]] = None,
    ):
        self.quantity = quantity
        self.unit = unit if unit is not None else ""
        self.per_atom = per_atom
        self._gradients = set(gradients) if gradients is not None else set()

    @property
    def gradients(self) -> List[str]:
        """Sorted and unique list of gradient names."""
        return sorted(self._gradients)

    @gradients.setter
    def gradients(self, value: List[str]):
        self._gradients = set(value)

    def __repr__(self):
        return (
            f"TargetInfo(quantity={self.quantity!r}, unit={self.unit!r}, "
            f"per_atom={self.per_atom!r}, gradients={self.gradients!r})"
        )

    def __eq__(self, other):
        if not isinstance(other, TargetInfo):
            raise NotImplementedError(
                "Comparison between a TargetInfo instance and a "
                f"{type(other).__name__} instance is not implemented."
            )
        return (
            self.quantity == other.quantity
            and self.unit == other.unit
            and self.per_atom == other.per_atom
            and self._gradients == other._gradients
        )

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

        self.gradients = self.gradients + other.gradients

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


class DatasetInfo:
    """A class that contains information about datasets.

    This class is used to communicate additional dataset details to the
    training functions of the individual models.

    :param length_unit: Unit of length used in the dataset. Examples are ``"angstrom"``
        or ``"nanometer"``.
    :param atomic_types: List containing all integer atomic types present in the
        dataset. ``atomic_types`` will be stored as a sorted list of **unique** atomic
        types.
    :param targets: Information about targets in the dataset.
    """

    def __init__(
        self, length_unit: str, atomic_types: List[int], targets: TargetInfoDict
    ):
        self.length_unit = length_unit if length_unit is not None else ""
        self._atomic_types = set(atomic_types)
        self.targets = targets

    @property
    def atomic_types(self) -> List[int]:
        """Sorted list of unique integer atomic types."""
        return sorted(self._atomic_types)

    @atomic_types.setter
    def atomic_types(self, value: List[int]):
        self._atomic_types = set(value)

    def __repr__(self):
        return (
            f"DatasetInfo(length_unit={self.length_unit!r}, "
            f"atomic_types={self.atomic_types!r}, targets={self.targets!r})"
        )

    def __eq__(self, other):
        if not isinstance(other, DatasetInfo):
            raise NotImplementedError(
                "Comparison between a DatasetInfo instance and a "
                f"{type(other).__name__} instance is not implemented."
            )
        return (
            self.length_unit == other.length_unit
            and self._atomic_types == other._atomic_types
            and self.targets == other.targets
        )

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

        self.atomic_types = self.atomic_types + other.atomic_types
        self.targets.update(other.targets)

    def union(self, other: "DatasetInfo") -> "DatasetInfo":
        """Return the union of this instance with ``other``."""
        new = self.copy()
        new.update(other)
        return new


def get_stats(dataset: Union[Dataset, Subset], dataset_info: DatasetInfo) -> str:
    """Returns the statistics of a dataset or subset as a string."""

    dataset_len = len(dataset)
    stats = f"Dataset containing {dataset_len} structures"
    if dataset_len == 0:
        return stats

    # target_names will be used to store names of the targets,
    # along with their gradients
    target_names = []
    for key, tensor_map in dataset[0]._asdict().items():
        if key == "system":
            continue
        target_names.append(key)
        gradients_list = tensor_map.block(0).gradients_list()
        for gradient in gradients_list:
            target_names.append(f"{key}_{gradient}_gradients")

    sums = {key: 0.0 for key in target_names}
    sums_of_squares = {key: 0.0 for key in target_names}
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
            sums_of_squares[key] += sum((tensor**2).sum() for tensor in tensors)
            n_elements[key] += sum(tensor.numel() for tensor in tensors)
    means = {key: sums[key] / n_elements[key] for key in target_names}
    means_of_squares = {
        key: sums_of_squares[key] / n_elements[key] for key in target_names
    }
    stds = {
        key: (means_of_squares[key] - means[key] ** 2) ** 0.5 for key in target_names
    }

    # Find units
    units = {}
    for key in target_names:
        # Gets the units of an output
        if key.endswith("_gradients"):
            # handling <base_name>_<gradient_name>_gradients
            base_name = key[:-10]
            gradient_name = base_name.split("_")[-1]
            base_name = base_name.replace(f"_{gradient_name}", "")
            base_unit = dataset_info.targets[base_name].unit
            unit = get_gradient_units(
                base_unit, gradient_name, dataset_info.length_unit
            )
        else:
            unit = dataset_info.targets[key].unit
        units[key] = unit

    stats += "\n    Mean and standard deviation of targets:"
    for key in target_names:
        stats += (
            f"\n    - {to_external_name(key, dataset_info.targets)}: "  # type: ignore
            + f"\n      - mean {means[key]:.4g}"
            + (f" {units[key]}" if units[key] != "" else "")
            + f"\n      - std  {stds[key]:.4g}"
            + (f" {units[key]}" if units[key] != "" else "")
        )

    return stats


def get_atomic_types(datasets: Union[Dataset, List[Dataset]]) -> List[int]:
    """List of all atomic types present in a dataset or list of datasets.

    :param datasets: the dataset, or list of datasets
    :returns: sorted list of all atomic types present in the datasets
    """

    if not isinstance(datasets, list):
        datasets = [datasets]

    types = set()
    for dataset in datasets:
        for index in range(len(dataset)):
            system = dataset[index]["system"]
            types.update(set(system.types.tolist()))

    return sorted(types)


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
            # system not needed
            target_names += [key for key in sample._asdict().keys() if key != "system"]

    return sorted(set(target_names))


def collate_fn(batch: List[Dict[str, Any]]) -> Tuple[List, Dict[str, TensorMap]]:
    """
    Wraps `group_and_join` to
    return the data fields as a list of systems, and a dictionary of nameed
    targets.
    """

    collated_targets = group_and_join(batch)
    collated_targets = collated_targets._asdict()
    systems = collated_targets.pop("system")
    return systems, collated_targets


def check_datasets(train_datasets: List[Dataset], val_datasets: List[Dataset]):
    """Check that the training and validation sets are compatible with one another

    Although these checks will not fit all use cases, most models would be expected
    to be able to use this function.

    :param train_datasets: A list of training datasets to check.
    :param val_datasets: A list of validation datasets to check
    :raises TypeError: If the ``dtype`` within the datasets are inconsistent.
    :raises ValueError: If the `val_datasets` has a target that is not present in
        the ``train_datasets``.
    :raises ValueError: If the training or validation set contains chemical species
        or targets that are not present in the training set
    """
    # Check that system `dtypes` are consistent within datasets
    desired_dtype = None
    for train_dataset in train_datasets:
        if len(train_dataset) == 0:
            continue

        actual_dtype = train_dataset[0].system.positions.dtype
        if desired_dtype is None:
            desired_dtype = actual_dtype

        if actual_dtype != desired_dtype:
            raise TypeError(
                "`dtype` between datasets is inconsistent, "
                f"found {desired_dtype} and {actual_dtype} in training datasets"
            )

    for val_dataset in val_datasets:
        if len(val_dataset) == 0:
            continue

        actual_dtype = val_dataset[0].system.positions.dtype

        if desired_dtype is None:
            desired_dtype = actual_dtype

        if actual_dtype != desired_dtype:
            raise TypeError(
                "`dtype` between datasets is inconsistent, "
                f"found {desired_dtype} and {actual_dtype} in validation datasets"
            )

    # Get all targets in the training and validation sets:
    train_targets = get_all_targets(train_datasets)
    val_targets = get_all_targets(val_datasets)

    # Check that the validation sets do not have targets that are not in the
    # training sets:
    for target in val_targets:
        if target not in train_targets:
            raise ValueError(
                f"The validation dataset has a target ({target}) that is not present "
                "in the training dataset."
            )
    # Get all the species in the training and validation sets:
    all_train_species = get_atomic_types(train_datasets)
    all_val_species = get_atomic_types(val_datasets)

    # Check that the validation sets do not have species that are not in the
    # training sets:
    for species in all_val_species:
        if species not in all_train_species:
            raise ValueError(
                f"The validation dataset has a species ({species}) that is not in the "
                "training dataset. This could be a result of a random train/validation "
                "split. You can avoid this by providing a validation dataset manually."
            )


def _train_test_random_split(
    train_dataset: Dataset,
    train_size: float,
    test_size: float,
) -> List[Dataset]:
    if train_size <= 0:
        raise ValueError("Fraction of the train set is smaller or equal to 0!")

    # normalize the sizes
    size_sum = train_size + test_size
    train_size /= size_sum
    test_size /= size_sum

    # find number of samples in the train and test sets
    test_len = math.floor(len(train_dataset) * test_size)
    if test_len == 0:
        warnings.warn(
            "Requested dataset of zero length. This dataset will be empty.",
            UserWarning,
            stacklevel=2,
        )
    train_len = len(train_dataset) - test_len
    if train_len == 0:
        raise ValueError("No samples left in the training set.")

    # find train, test indices
    indices = list(range(len(train_dataset)))
    np.random.shuffle(indices)
    train_indices = indices[:train_len]
    test_indices = indices[train_len:]

    return [
        Subset(train_dataset, train_indices),
        Subset(train_dataset, test_indices),
    ]
