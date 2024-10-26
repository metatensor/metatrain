import math
import warnings
from typing import Any, Dict, List, Tuple, Union

import metatensor.torch
import numpy as np
from metatensor.learn.data import Dataset, group_and_join
from metatensor.torch import TensorMap
from torch.utils.data import Subset

from ..external_naming import to_external_name
from ..units import get_gradient_units


class TargetInfo:
    """A class that contains information about a target.

    :param quantity: The physical quantity of the target (e.g., "energy").
    :param layout: The layout of the target, as a ``TensorMap`` with 0 samples.
        This ``TensorMap`` will contain important information such as the names of
        the ``samples``, as well as the ``components`` and ``properties`` of the
        target and their gradients.
    :param unit: The unit of the target. If :py:obj:`None` the ``unit`` will be set to
        an empty string ``""``.
    """

    def __init__(
        self,
        quantity: str,
        layout: TensorMap,
        unit: Union[None, str] = "",
    ):
        # one of these will be set to True inside the _check_layout method
        self.is_scalar = False
        self.is_cartesian = False
        self.is_spherical = False

        self._check_layout(layout)

        self.quantity = quantity
        self.layout = layout
        self.unit = unit if unit is not None else ""

    @property
    def gradients(self) -> List[str]:
        """Sorted and unique list of gradient names."""
        if self.is_scalar:
            return sorted(self.layout.block().gradients_list())
        else:
            return []

    @property
    def per_atom(self) -> bool:
        """Whether the target is per atom."""
        return "atom" in self.layout.block(0).samples.names

    def __repr__(self):
        return (
            f"TargetInfo(quantity={self.quantity!r}, unit={self.unit!r}, "
            f"layout={self.layout!r})"
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
            and metatensor.torch.equal(self.layout, other.layout)
        )

    def _check_layout(self, layout: TensorMap) -> None:
        """Check that the layout is a valid layout."""

        # examine basic properties of all blocks
        for block in layout.blocks():
            for sample_name in block.samples.names:
                if sample_name not in ["system", "atom"]:
                    raise ValueError(
                        "The layout ``TensorMap`` of a target should only have samples "
                        "named 'system' or 'atom', but found "
                        f"'{sample_name}' instead."
                    )
            if len(block.values) != 0:
                raise ValueError(
                    "The layout ``TensorMap`` of a target should have 0 "
                    f"samples, but found {len(block.values)} samples."
                )

        # examine the components of the first block to decide whether this is
        # a scalar, a Cartesian tensor or a spherical tensor

        if len(layout) == 0:
            raise ValueError(
                "The layout ``TensorMap`` of a target should have at least one "
                "block, but found 0 blocks."
            )
        components_first_block = layout.block(0).components
        if len(components_first_block) == 0:
            self.is_scalar = True
        elif components_first_block[0].names[0].startswith("xyz"):
            self.is_cartesian = True
        elif (
            len(components_first_block) == 1
            and components_first_block[0].names[0] == "o3_mu"
        ):
            self.is_spherical = True
        else:
            raise ValueError(
                "The layout ``TensorMap`` of a target should be "
                "either scalars, Cartesian tensors or spherical tensors. The type of "
                "the target could not be determined."
            )

        if self.is_scalar:
            if layout.keys.names != ["_"]:
                raise ValueError(
                    "The layout ``TensorMap`` of a scalar target should have "
                    "a single key sample named '_'."
                )
            if len(layout.blocks()) != 1:
                raise ValueError(
                    "The layout ``TensorMap`` of a scalar target should have "
                    "a single block."
                )
            gradients_names = layout.block(0).gradients_list()
            for gradient_name in gradients_names:
                if gradient_name not in ["positions", "strain"]:
                    raise ValueError(
                        "Only `positions` and `strain` gradients are supported for "
                        "scalar targets. "
                        f"Found '{gradient_name}' instead."
                    )
        if self.is_cartesian:
            if layout.keys.names != ["_"]:
                raise ValueError(
                    "The layout ``TensorMap`` of a Cartesian tensor target should have "
                    "a single key sample named '_'."
                )
            if len(layout.blocks()) != 1:
                raise ValueError(
                    "The layout ``TensorMap`` of a Cartesian tensor target should have "
                    "a single block."
                )
            if len(layout.block(0).gradients_list()) > 0:
                raise ValueError(
                    "Gradients of Cartesian tensor targets are not supported."
                )

        if self.is_spherical:
            if layout.keys.names != ["o3_lambda", "o3_sigma"]:
                raise ValueError(
                    "The layout ``TensorMap`` of a spherical tensor target "
                    "should have  two keys named 'o3_lambda' and 'o3_sigma'."
                    f"Found '{layout.keys.names}' instead."
                )
            for key, block in layout.items():
                o3_lambda, o3_sigma = int(key.values[0].item()), int(
                    key.values[1].item()
                )
                if o3_sigma not in [-1, 1]:
                    raise ValueError(
                        "The layout ``TensorMap`` of a spherical tensor target should "
                        "have a key sample 'o3_sigma' that is either -1 or 1."
                        f"Found '{o3_sigma}' instead."
                    )
                if o3_lambda < 0:
                    raise ValueError(
                        "The layout ``TensorMap`` of a spherical tensor target should "
                        "have a key sample 'o3_lambda' that is non-negative."
                        f"Found '{o3_lambda}' instead."
                    )
                components = block.components
                if len(components) != 1:
                    raise ValueError(
                        "The layout ``TensorMap`` of a spherical tensor target should "
                        "have a single component."
                    )
                if len(components[0]) != 2 * o3_lambda + 1:
                    raise ValueError(
                        "Each ``TensorBlock`` of a spherical tensor target should have "
                        "a component with 2*o3_lambda + 1 elements."
                        f"Found '{len(components[0])}' elements instead."
                    )
                if len(block.gradients_list()) > 0:
                    raise ValueError(
                        "Gradients of spherical tensor targets are not supported."
                    )


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
        self, length_unit: str, atomic_types: List[int], targets: Dict[str, TargetInfo]
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

        intersecting_target_keys = self.targets.keys() & other.targets.keys()
        for key in intersecting_target_keys:
            if self.targets[key] != other.targets[key]:
                raise ValueError(
                    f"Can't update DatasetInfo with different target information for "
                    f"target '{key}': {self.targets[key]} != {other.targets[key]}"
                )
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
