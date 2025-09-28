import math
import multiprocessing
import os
import warnings
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

import metatensor.torch
import numpy as np
import torch
from metatensor.learn.data import Dataset, group_and_join
from metatensor.learn.data._namedtuple import namedtuple
from metatensor.torch import TensorMap, load_buffer, save_buffer
from metatomic.torch import load_system, load_system_buffer, save
from metatomic.torch import save_buffer as save_system_buffer
from metatensor.torch import Labels, TensorBlock, TensorMap, load_buffer
from metatomic.torch import System, load_system
from omegaconf import DictConfig
from torch.utils.data import Dataset as TorchDataset
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


class SystemWrapper:
    """A wrapper for ``metatomic.torch.System`` that makes it pickle-compatible."""

    def __init__(self, system):
        self.system = system

    def __getstate__(self):
        state = BytesIO()
        save(state, self.system)
        return state

    def __setstate__(self, state):
        self.system = load_system(state)


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
        callables: Optional[List[Callable]] = None,
        join_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.target_keys: Set[str] = set(target_keys)
        self.callables: List[Callable] = callables if callables is not None else []
        self.join_kwargs: Dict[str, Any] = join_kwargs or {
            "remove_tensor_name": True,
            "different_keys": "union",
        }

    def __call__(
        self,
        batch: List[Dict[str, Any]],
    ):
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

        for callable in self.callables:
            systems, targets, extra = callable(systems, targets, extra)

        target_names = list(targets.keys())
        extra_names = list(extra.keys())

        system_buffers = [save_system_buffer(s) for s in systems]
        target_buffers = [save_buffer(targets[name]) for name in target_names]
        extra_buffers = [save_buffer(extra[name]) for name in extra_names]

        system_sizes = [len(b) for b in system_buffers]
        target_sizes = [len(b) for b in target_buffers]
        extra_sizes = [len(b) for b in extra_buffers]

        blob = torch.concatenate(system_buffers + target_buffers + extra_buffers)

        return blob, system_sizes, target_names, target_sizes, extra_names, extra_sizes


def unpack_batch(batch):
    blob, system_sizes, target_names, target_sizes, extra_names, extra_sizes = batch

    all_buffers = torch.split(blob, system_sizes + target_sizes + extra_sizes)
    systems = all_buffers[: len(system_sizes)]
    targets = {
        name: buf
        for name, buf in zip(
            target_names,
            all_buffers[len(system_sizes) : len(system_sizes) + len(target_names)],
        )
    }
    extra_data = {
        name: buf
        for name, buf in zip(
            extra_names, all_buffers[len(system_sizes) + len(target_names) :]
        )
    }

    systems = [load_system_buffer(s) for s in systems]
    targets = {key: load_buffer(t) for key, t in targets.items()}
    extra_data = {key: load_buffer(t) for key, t in extra_data.items()}
    return systems, targets, extra_data


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
                    target_key, target, add_position_gradients, add_strain_gradients
                )
                _check_tensor_map_metadata(tensor_map, target_info.layout)
                target_info_dict[target_key] = target_info
            else:
                target_info = get_generic_target_info(target_key, target)
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


def get_num_workers() -> int:
    """Gets a good number of workers for data loading."""

    # len(os.sched_getaffinity(0)) detects thread counts set by slurm,
    # multiprocessing.cpu_count() doesn't but is more portable
    if hasattr(os, "sched_getaffinity"):
        num_threads = min(len(os.sched_getaffinity(0)), multiprocessing.cpu_count())
    else:
        num_threads = multiprocessing.cpu_count()

    reserve = 4  # main training process, NCCL, GPU driver, loggers, ...
    cap = 8  # above this can overwhelm the filesystem

    # can't go below 0, in that case the main training process will handle data loading
    num_workers = max(0, min(num_threads - reserve, cap))

    return num_workers


def memmap_collate_fn(batch):
    non_system_keys = [key for key in batch[0].keys() if key != "system"]
    systems = [sample["system"] for sample in batch]
    targets = {k: [] for k in non_system_keys}
    for sample in batch:
        for key in non_system_keys:
            targets[key].append(sample[key])
    targets = {k: metatensor.torch.join(v, "samples") for k, v in targets.items()}
    return systems, targets, {}


class MemmapArray:
    """Small helper to reopen np.memmap lazily in each worker."""

    def __init__(self, path, shape, dtype, mode="r"):
        self.path = str(path)
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.mode = mode
        self._mm = None

    def _ensure_open(self):
        if self._mm is None:
            self._mm = np.memmap(
                self.path, dtype=self.dtype, mode=self.mode, shape=self.shape
            )

    def __getitem__(self, idx):
        self._ensure_open()
        return self._mm[idx]

    def close(self):
        if self._mm is not None:
            # np.memmap closes when deleted; explicit close via _mmap isn't public.
            self._mm._mmap.close()
            self._mm = None


class MemmapDataset(TorchDataset):
    """A class representing a dataset stored as a set of memory-mapped arrays.

    This dataset supports arbitrary scalar and cartesian vector/tensor targets, but
    not spherical tensors. Virials of energy targets are not supported in this type of
    dataset (stresses can be used instead to achieve the same goal).

    The dataset is stored in a directory, where the dataset is stored in a set of
    memory-mapped numpy arrays. These are:
    - N.npy: total number of structures in the dataset. Shape: (1,).
    - n.npy: cumulative number of atoms per structure. n[-1] therefore corresponds to
        the total number of atoms in the dataset. Shape: (N+1,).
    - x.bin: atomic positions of all atoms in the dataset, concatenated. Shape:
        (n[-1], 3).
    - a.bin: atomic types of all atoms in the dataset, concatenated. Shape: (n[-1],).
    - c.bin: cell matrices of all structures in the dataset. Shape: (N, 3, 3).
    - <key>.bin: target values for each structure or atom, depending on the
        whether the target is defined per atom or per structure.
        Shape: (N, ..., num_subtargets) if per-structures or
        (n[-1], ..., num_subtargets) if per-atom, where the
        ... depends on the type of target (scalar, vector, tensor, etc.). <key> can
        then be used in the "key" section of targets in metatrain input files to read
        the target(s).

    :param path: Path to the directory containing the dataset.
    :param target_options: Dictionary containing the target configurations, in the
        format corresponding to metatrain yaml input files.
    """

    def __init__(self, path: str | Path, target_options: Dict[str, Any]):
        path = Path(path)
        self.target_config = target_options
        self.sample_class = namedtuple(
            "Sample", ["system"] + list(self.target_config.keys())
        )

        # Information about the structures
        self.N = np.load(path / "N.npy")
        self.n = np.load(path / "n.npy")
        self.x = MemmapArray(path / "x.bin", (self.n[-1], 3), "float32", mode="r")
        self.a = MemmapArray(path / "a.bin", (self.n[-1],), "int32", mode="r")
        self.c = MemmapArray(path / "c.bin", (self.N, 3, 3), "float32", mode="r")

        # Register arrays pointing to the targets
        self.target_arrays = {}
        for target_key, single_target_options in target_options.items():
            data_key = single_target_options["key"]
            number_of_samples = (
                self.n[-1] if single_target_options["per_atom"] else self.N
            )
            number_of_properties = single_target_options["num_subtargets"]
            if single_target_options["type"] == "scalar":
                self.target_arrays[target_key] = MemmapArray(
                    path / f"{data_key}.bin",
                    (number_of_samples, number_of_properties),
                    "float32",
                    mode="r",
                )
                if (
                    single_target_options["quantity"] == "energy"
                    and not single_target_options["per_atom"]
                    and single_target_options["num_subtargets"] == 1
                ):
                    # energy target: look into potential gradients
                    if single_target_options["forces"]:
                        self.target_arrays[f"{target_key}_forces"] = MemmapArray(
                            path / f"{single_target_options['forces']['key']}.bin",
                            (self.n[-1], 3, 1),
                            "float32",
                            mode="r",
                        )
                    if single_target_options["stress"]:
                        self.target_arrays[f"{target_key}_stress"] = MemmapArray(
                            path / f"{single_target_options['stress']['key']}.bin",
                            (self.N, 3, 3, 1),
                            "float32",
                            mode="r",
                        )
                    if single_target_options["virial"]:
                        raise ValueError(
                            "Virial targets are not supported in MemmapDataset."
                        )
            elif isinstance(single_target_options["type"], DictConfig) or isinstance(
                single_target_options["type"], Dict
            ):
                if "spherical" in single_target_options["type"]:
                    raise ValueError(
                        "Spherical targets are not supported in MemmapDataset."
                    )
                else:  # cartesian
                    n_components = single_target_options["type"]["cartesian"]["rank"]
                    shape = (
                        (number_of_samples,)
                        + (3,) * n_components
                        + (number_of_properties,)
                    )
                    self.target_arrays[target_key] = MemmapArray(
                        path / f"{data_key}.bin", shape, "float32", mode="r"
                    )
            else:
                raise ValueError(
                    f"Unsupported target configuration: {single_target_options}"
                )

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        a = torch.tensor(self.a[self.n[i] : self.n[i + 1]], dtype=torch.int32)
        x = torch.tensor(self.x[self.n[i] : self.n[i + 1]], dtype=torch.float64)
        c = torch.tensor(self.c[i], dtype=torch.float64)

        system = System(
            positions=x,
            types=a,
            cell=c,
            pbc=torch.logical_not(torch.all(c == 0.0, dim=1)),
        )

        target_dict = {}
        for target_key, target_options in self.target_config.items():
            target_array = self.target_arrays[target_key]
            is_per_atom = target_array.shape[0] == (self.n[-1])
            if is_per_atom:
                samples = Labels(
                    names=["system", "atom"],
                    values=torch.tensor(
                        [[i, j] for j in range(self.n[i], self.n[i + 1])],
                        dtype=torch.int32,
                    ),
                )
            else:
                samples = Labels(
                    names=["system"],
                    values=torch.tensor([[i]], dtype=torch.int32),
                )
            if len(target_array.shape) > 3:
                # Cartesian tensor with rank > 1
                n_components = len(target_array.shape) - 2
                components = [
                    Labels.range(f"xyz_{d + 1}", 3) for d in range(n_components)
                ]
            elif len(target_array.shape) == 3:
                # Cartesian vector
                components = [Labels.range("xyz", 3)]
            else:
                # Scalar
                components = []

            target_block = TensorBlock(
                values=torch.tensor(
                    target_array[None, i]
                    if not is_per_atom
                    else target_array[self.n[i] : self.n[i + 1]],
                    dtype=torch.float64,
                ),
                samples=samples,
                components=components,
                properties=Labels.range(target_key, target_array.shape[-1]),
            )

            # handle energy gradients
            if (
                target_options["quantity"] == "energy"
                and not target_options["per_atom"]
                and target_options["num_subtargets"] == 1
            ):
                if target_options["forces"]:
                    f = torch.tensor(
                        self.target_arrays[f"{target_key}_forces"][
                            self.n[i] : self.n[i + 1]
                        ],
                        dtype=torch.float64,
                    )
                    target_block.add_gradient(
                        "positions",
                        TensorBlock(
                            values=-f,
                            samples=Labels(
                                names=["sample", "atom"],
                                values=torch.tensor(
                                    [[0, j] for j in range(len(a))], dtype=torch.int32
                                ),
                            ),
                            components=[Labels.range("xyz", 3)],
                            properties=Labels.range("energy", 1),
                        ),
                    )
                if target_options["stress"]:
                    s = torch.tensor(
                        self.target_arrays[f"{target_key}_stress"][None, i],
                        dtype=torch.float64,
                    )
                    target_block.add_gradient(
                        "strain",
                        TensorBlock(
                            values=(s * torch.abs(torch.det(c))),
                            samples=Labels(
                                names=["sample"],
                                values=torch.tensor([[0]], dtype=torch.int32),
                            ),
                            components=[
                                Labels.range("xyz_1", 3),
                                Labels.range("xyz_2", 3),
                            ],
                            properties=Labels.range("energy", 1),
                        ),
                    )

            target_tensormap = TensorMap(
                keys=Labels.single(),
                blocks=[target_block],
            )
            target_dict[target_key] = target_tensormap

        sample = self.sample_class(**{"system": system, **target_dict})
        return sample

    def get_target_info(self) -> Dict[str, TargetInfo]:
        """
        Get information about the targets in the dataset.
        """
        target_info_dict = {}
        for target_key, target in self.target_config.items():
            is_energy = (
                (target["quantity"] == "energy")
                and (not target["per_atom"])
                and target["num_subtargets"] == 1
                and target["type"] == "scalar"
            )
            tensor_map = self[0][target_key]
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
