from pathlib import Path
from typing import Dict, List, Tuple

from metatensor.torch import TensorMap
from omegaconf import DictConfig

from .dataset import Dataset, DiskDataset, MemmapDataset
from .readers import (
    read_extra_data,
    read_sample_weights,
    read_systems,
    read_targets,
)
from .readers.readers import _target_defines_sample_weights
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

    requests_sample_weights = any(
        _target_defines_sample_weights(entry)
        for entry in options.get("targets", {}).values()
    )

    if options["systems"]["read_from"].endswith(".zip"):  # disk dataset
        if requests_sample_weights:
            raise NotImplementedError(
                "'sample_weight_key' is not supported for disk (.zip) datasets. "
                "Provide the weights as an 'extra_data' field named "
                "'<target>_weights' instead."
            )
        dataset = DiskDataset(
            options["systems"]["read_from"],
            fields=[*options["targets"], *options.get("extra_data", {})],
        )
        target_info_dictionary = dataset.get_target_info(options["targets"])
        extra_data_info_dictionary = dataset.get_target_info(
            options.get("extra_data", {}), is_extra_data=True
        )
    elif Path(options["systems"]["read_from"]).is_dir():  # memmap dataset
        if requests_sample_weights:
            raise NotImplementedError(
                "'sample_weight_key' is not supported for memmap datasets. "
                "Provide the weights as an 'extra_data' field named "
                "'<target>_weights' instead."
            )
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

        # Read per-sample loss weights (sample_weight_key) into extra_data. The weight
        # TensorMaps mirror the structure of their targets and are stored under the key
        # "<target>_weights", where the weighted loss functions pick them up.
        if requests_sample_weights:
            sample_weights = read_sample_weights(options["targets"], targets)
            intersecting_keys = extra_data.keys() & sample_weights.keys()
            if intersecting_keys:
                raise ValueError(
                    f"Sample weight keys {intersecting_keys} overlap with extra_data "
                    "keys. The '<target>_weights' names are reserved when using "
                    "'sample_weight_key'."
                )
            for weights_key, weights_tensor_maps in sample_weights.items():
                target_name = weights_key[: -len("_weights")]
                extra_data[weights_key] = weights_tensor_maps
                extra_data_info_dictionary[weights_key] = TargetInfo(
                    layout=target_info_dictionary[target_name].layout,
                    quantity="",
                    unit="",
                )

        dataset = Dataset.from_dict({"system": systems, **targets, **extra_data})

    return dataset, target_info_dictionary, extra_data_info_dictionary
