import math
import os
import warnings
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from metatensor.learn.data import Dataset, group_and_join
from metatensor.learn.data._namedtuple import namedtuple
from metatensor.torch import TensorMap, load_buffer
from metatomic.torch import load_system
from omegaconf import DictConfig
from torch.utils.data import Subset

from metatrain.utils.data.readers.metatensor import (
    _check_tensor_map_metadata,
    _empty_tensor_map_like,
)
from metatrain.utils.data.target_info import (
    TargetInfo,
    get_energy_target_info,
    get_generic_target_info,
)
from metatrain.utils.external_naming import to_external_name
from metatrain.utils.units import get_gradient_units


def _set(values: List[int]) -> List[int]:
    """This function just does `list(set(values))`.

    But set is not torchscript compatible, so we do it manually.
    """
    unique_values: List[int] = []
    for at_type in values:
        found = False
        for seen in unique_values:
            if seen == at_type:
                found = True
                break
        if not found:
            unique_values.append(at_type)

    return unique_values


class DatasetInfo:
    """A class that contains information about datasets.

    This class is used to communicate additional dataset details to the
    training functions of the individual models.

    :param length_unit: Unit of length used in the dataset. Examples are ``"angstrom"``
        or ``"nanometer"``. If None, the unit will be set to the empty string.
    :param atomic_types: List containing all integer atomic types present in the
        dataset. ``atomic_types`` will be stored as a sorted list of **unique** atomic
        types.
    :param targets: Information about targets in the dataset.
    :param extra_data: Optional dictionary containing additional data that is not
        used as a target, but is still relevant to the dataset.
    """

    def __init__(
        self,
        length_unit: Optional[str],
        atomic_types: List[int],
        targets: Dict[str, TargetInfo],
        extra_data: Optional[Dict[str, TargetInfo]] = None,
    ):
        self.length_unit = length_unit if length_unit is not None else ""
        self._atomic_types = _set(atomic_types)
        self.targets = targets
        self.extra_data: Dict[str, TargetInfo] = (
            extra_data if extra_data is not None else {}
        )

    @property
    def atomic_types(self) -> List[int]:
        """Sorted list of unique integer atomic types."""
        return sorted(self._atomic_types)

    @atomic_types.setter
    def atomic_types(self, value: List[int]):
        self._atomic_types = _set(value)

    @property
    def device(self) -> Optional[torch.device]:
        """Return the device where the tensors of DatasetInfo are located.

        This function only checks the device of the first target
        and assumes that all targets and extra data are on the same device.
        This is guaranteed if the ``to()`` method has been used to move
        the DatasetInfo to a specific device.
        """
        if len(self.targets) == 0:
            return None
        first_target = list(self.targets.values())[0]
        return first_target.device

    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ) -> "DatasetInfo":
        """Return a copy with all tensors moved to the device and dtype."""
        new = self.copy()
        for key, target_info in new.targets.items():
            new.targets[key] = target_info.to(device=device, dtype=dtype)
        for key, extra_data in new.extra_data.items():
            new.extra_data[key] = extra_data.to(device=device, dtype=dtype)
        return new

    def __repr__(self):
        return "DatasetInfo(length_unit={!r}, atomic_types={!r}, targets={!r})".format(
            self.length_unit, self.atomic_types, self.targets
        )

    def __eq__(self, other):
        if not isinstance(other, DatasetInfo):
            return False
        return (
            self.length_unit == other.length_unit
            and self._atomic_types == other._atomic_types
            and self.targets == other.targets
            and self.extra_data == other.extra_data
        )

    def copy(self) -> "DatasetInfo":
        """Return a shallow copy of the DatasetInfo."""
        return DatasetInfo(
            length_unit=self.length_unit,
            atomic_types=self.atomic_types.copy(),
            targets=self.targets.copy(),
            extra_data=self.extra_data.copy(),
        )

    @torch.jit.unused
    def update(self, other: "DatasetInfo") -> None:
        """Update this instance with the union of itself and ``other``.

        :raises ValueError: If the ``length_units`` are different.
        """
        if self.length_unit != other.length_unit:
            raise ValueError(
                "Can't update DatasetInfo with a different `length_unit`: "
                f"('{self.length_unit}' != '{other.length_unit}')"
            )

        self.atomic_types = self.atomic_types + other.atomic_types

        intersecting_target_keys = self.targets.keys() & other.targets.keys()
        for key in intersecting_target_keys:
            if not self.targets[key].is_compatible_with(other.targets[key]):
                raise ValueError(
                    f"Can't update DatasetInfo with different target information for "
                    f"target '{key}': {self.targets[key]} is not compatible with "
                    f"{other.targets[key]}. If the units, quantity and keys of the two "
                    "targets are the same, this must be due to a mismatch in the "
                    "internal metadata of the layout."
                )
        self.targets.update(other.targets)

        intersecting_extra_data_keys = self.extra_data.keys() & other.extra_data.keys()
        for key in intersecting_extra_data_keys:
            if not self.extra_data[key].is_compatible_with(other.extra_data[key]):
                raise ValueError(
                    f"Can't update DatasetInfo with different extra data information "
                    f"for key '{key}': {self.extra_data[key]} is not compatible with "
                    f"{other.extra_data[key]}. If the units, quantity and keys of the "
                    "two extra data dictionaries are the same, this must be due to a "
                    "mismatch in the internal metadata of the layout."
                )
        self.extra_data.update(other.extra_data)

    def union(self, other: "DatasetInfo") -> "DatasetInfo":
        """Return the union of this instance with ``other``."""
        new = self.copy()
        new.update(other)
        return new

    @torch.jit.unused
    def __setstate__(self, state):
        """
        Custom ``__setstate__`` to allow loading old checkpoints where ``extra_data`` is
        missing.
        """
        self.length_unit = state["length_unit"]
        self._atomic_types = state["_atomic_types"]
        self.targets = state["targets"]
        self.extra_data = state.get("extra_data", {})


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
                # The name is <basename>_<gradname>_gradients
                original_key = "_".join(key.split("_")[:-2])
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
        if key not in dataset_info.targets:
            continue
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

    # TODO: add extra data statistics?

    stats += "\n    Mean and standard deviation of targets:"
    for key in target_names:
        if key not in means or key not in units or key not in stds:
            continue
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
        for sample in dataset:
            system = sample["system"]
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


class CollateFn:
    def __init__(
        self,
        target_keys: List[str],
        join_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.target_keys: Set[str] = set(target_keys)
        self.join_kwargs: Dict[str, Any] = join_kwargs or {
            "remove_tensor_name": True,
            "different_keys": "union",
        }

    def __call__(
        self,
        batch: List[Dict[str, Any]],
    ) -> Tuple[
        Any,  # systems
        Dict[str, TensorMap],  # targets
        Dict[str, TensorMap],  # extra data
    ]:
        # group & join
        collated = group_and_join(batch, join_kwargs=self.join_kwargs)
        data = collated._asdict()

        # pull off systems
        systems = data.pop("system")

        # split into targets vs extra data
        targets: Dict[str, TensorMap] = {}
        extra: Dict[str, TensorMap] = {}

        for key, value in data.items():
            if key in self.target_keys:
                targets[key] = value
            else:
                extra[key] = value

        return systems, targets, extra


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


class DiskDataset(torch.utils.data.Dataset):
    """A class representing a dataset stored on disk.

    The dataset is stored in a zip file, where each sample is stored in a separate
    directory. The directory's name is the index of the sample (e.g. ``0/``), and the
    files in the directory are the system (``system.mta``) and the targets (each named
    ``<target_name>.mts``). These are ``metatomic.torch.System`` and
    ``metatensor.torch.TensorMap`` objects, respectively.

    Such a dataset can be created conveniently using the :py:class:`DiskDatasetWriter`
    class.

    :param path: Path to the zip file containing the dataset.
    :param fields: List of fields to read from the dataset.
        If None, all fields will be read.
    """

    def __init__(self, path: Union[str, Path], fields: Optional[List[str]] = None):
        self.zip_file = zipfile.ZipFile(path, "r")
        self._field_names = ["system"]
        # check that we have at least one sample:
        if "0/system.mta" not in self.zip_file.namelist():
            raise ValueError(
                "Could not find `0/system.mta` in the zip file. "
                "The dataset format might be wrong, or the dataset might be empty. "
                "Empty disk datasets are not supported."
            )
        for file_name in self.zip_file.namelist():
            if file_name.startswith("0/") and file_name.endswith(".mts"):
                self._field_names.append(file_name[2:-4])

        # Determine which fields are going to be read
        if fields is None:
            self._fields_to_read = self._field_names
        else:
            # Check that the requested fields are present in the dataset
            fields = ["system", *fields]
            missing_fields = set(fields) - set(self._field_names)
            if missing_fields:
                raise ValueError(
                    f"Fields {list(missing_fields)} were requested but "
                    "are not present in this disk dataset. "
                    f"Available fields: {self._field_names[1:]}"
                )
            self._fields_to_read = fields

        self._sample_class = namedtuple("Sample", self._fields_to_read)
        self._len = len([f for f in self.zip_file.namelist() if f.endswith(".mta")])

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        system_and_targets = []
        for field_name in self._fields_to_read:
            if field_name == "system":
                with self.zip_file.open(f"{index}/system.mta", "r") as file:
                    system = load_system(file)
                    system_and_targets.append(system)
            else:
                with self.zip_file.open(f"{index}/{field_name}.mts", "r") as file:
                    numpy_buffer = np.load(file)
                    tensor_buffer = torch.from_numpy(numpy_buffer)
                    tensor_map = load_buffer(tensor_buffer)
                    system_and_targets.append(tensor_map)
        return self._sample_class(*system_and_targets)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __del__(self):
        self.zip_file.close()

    def get_target_info(self, target_config: DictConfig) -> Dict[str, TargetInfo]:
        """
        Get information about the targets in the dataset.

        :param target_config: The user-provided (through the yaml file) target
            configuration.
        """
        target_info_dict = {}
        for target_key, target in target_config.items():
            is_energy = (
                (target["quantity"] == "energy")
                and (not target["per_atom"])
                and target["num_subtargets"] == 1
                and target["type"] == "scalar"
            )
            tensor_map = self[0][target_key]  # always > 0 samples, see above
            if is_energy:
                if len(tensor_map) != 1:
                    raise ValueError("Energy TensorMaps should have exactly one block.")
                add_position_gradients = tensor_map.block().has_gradient("positions")
                add_strain_gradients = tensor_map.block().has_gradient("strain")
                target_info = get_energy_target_info(
                    target, add_position_gradients, add_strain_gradients
                )
                _check_tensor_map_metadata(tensor_map, target_info.layout)
                target_info_dict[target_key] = target_info
            else:
                target_info = get_generic_target_info(target)
                _check_tensor_map_metadata(tensor_map, target_info.layout)
                # make sure that the properties of the target_info.layout also match the
                # actual properties of the tensor maps
                target_info.layout = _empty_tensor_map_like(tensor_map)
                target_info_dict[target_key] = target_info
        return target_info_dict


def _is_disk_dataset(dataset: Any) -> bool:
    # this also needs to detect if it's a ``torch.nn.utils.data.Subset`` object
    # with a ``DiskDataset`` object as its dataset, recursively
    if isinstance(dataset, DiskDataset):
        return True
    if isinstance(dataset, torch.utils.data.Subset):
        return _is_disk_dataset(dataset.dataset)
    return False


def _save_indices(
    train_indices: List[Optional[List[int]]],
    val_indices: List[Optional[List[int]]],
    test_indices: List[Optional[List[int]]],
    checkpoint_dir: Union[str, Path],
) -> None:
    # Save the indices of the training, validation, and test sets to the checkpoint
    # directory. This is useful for plotting errors and similar.

    # case 1: all indices are None (i.e. all datasets were user-provided explicitly)
    if all(indices is None for indices in train_indices):
        pass

    # case 2: there is only one dataset
    elif len(train_indices) == 1:  # val and test are the same length
        os.mkdir(os.path.join(checkpoint_dir, "indices/"))
        if train_indices[0] is not None:
            np.savetxt(
                os.path.join(checkpoint_dir, "indices/training.txt"),
                train_indices[0],
                fmt="%d",
            )
        if val_indices[0] is not None:
            np.savetxt(
                os.path.join(checkpoint_dir, "indices/validation.txt"),
                val_indices[0],
                fmt="%d",
            )
        if test_indices[0] is not None:
            np.savetxt(
                os.path.join(checkpoint_dir, "indices/test.txt"),
                test_indices[0],
                fmt="%d",
            )

    # case 3: there are multiple datasets
    else:
        os.mkdir(os.path.join(checkpoint_dir, "indices/"))
        for i, (train, val, test) in enumerate(
            zip(train_indices, val_indices, test_indices)
        ):
            if train is not None:
                np.savetxt(
                    os.path.join(checkpoint_dir, f"indices/training_{i}.txt"),
                    train,
                    fmt="%d",
                )
            if val is not None:
                np.savetxt(
                    os.path.join(checkpoint_dir, f"indices/validation_{i}.txt"),
                    val,
                    fmt="%d",
                )
            if test is not None:
                np.savetxt(
                    os.path.join(checkpoint_dir, f"indices/test_{i}.txt"),
                    test,
                    fmt="%d",
                )
