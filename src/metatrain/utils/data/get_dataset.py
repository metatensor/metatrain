from pathlib import Path
from typing import Dict, List, Tuple

from metatensor.torch import TensorMap
from omegaconf import DictConfig

from .dataset import Dataset, DiskDataset, MemmapDataset
from .readers import read_extra_data, read_systems, read_targets
from .target_info import TargetInfo


def get_dataset(
    options: DictConfig,
) -> Tuple[Dataset, Dict[str, TargetInfo], Dict[str, TargetInfo]]:
    """
    Gets a dataset given a configuration dictionary.

    The system and targets in the dataset are read from one or more
    files, as specified in ``options``.

    :param options: the configuration options for the dataset.
        This configuration dictionary must contain keys for both the
        systems and targets in the dataset.

    :return: A tuple containing a ``Dataset`` object and a
        ``Dict[str, TargetInfo]`` containing additional information (units,
        physical quantities, ...) on the targets in the dataset
    """

    extra_data_info_dictionary = {}

    if options["systems"]["read_from"].endswith(".zip"):  # disk dataset
        dataset = DiskDataset(
            options["systems"]["read_from"],
            fields=[*options["targets"], *options.get("extra_data", {})],
        )
        target_info_dictionary = dataset.get_target_info(options["targets"])
        if "extra_data" in options:
            extra_data_info_dictionary = dataset.get_target_info(options["extra_data"])
    elif Path(options["systems"]["read_from"]).is_dir():  # memmap dataset
        dataset = MemmapDataset(options["systems"]["read_from"], options["targets"])
        target_info_dictionary = dataset.get_target_info()
    else:
        systems = read_systems(
            filename=options["systems"]["read_from"],
            reader=options["systems"]["reader"],
        )
        targets, target_info_dictionary = read_targets(conf=options["targets"])
        extra_data: Dict[str, List[TensorMap]] = {}
        if "extra_data" in options:
            extra_data, extra_data_info_dictionary = read_extra_data(
                conf=options["extra_data"]
            )
            intersecting_keys = targets.keys() & extra_data.keys()
            if intersecting_keys:
                raise ValueError(
                    f"Extra data keys {intersecting_keys} overlap with target keys. "
                    "Please use unique keys for targets and extra data."
                )
        dataset = Dataset.from_dict({"system": systems, **targets, **extra_data})

    return dataset, target_info_dictionary, extra_data_info_dictionary
