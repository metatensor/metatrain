from typing import Tuple

from omegaconf import DictConfig

from .dataset import Dataset, TargetInfoDict
from .readers import read_systems, read_targets


def get_dataset(options: DictConfig) -> Tuple[Dataset, TargetInfoDict]:
    """
    Gets a dataset given a configuration dictionary.

    The system and targets in the dataset are read from one or more
    files, as specified in ``options``.

    :param options: the configuration options for the dataset.
        This configuration dictionary must contain keys for both the
        systems and targets in the dataset.

    :returns: A tuple containing a ``Dataset`` object and a
        ``TargetInfoDict`` containing additional information (units,
        physical quantities, ...) on the targets in the dataset
    """

    systems = read_systems(
        filename=options["systems"]["read_from"],
        reader=options["systems"]["reader"],
    )
    targets, target_info_dictionary = read_targets(conf=options["targets"])
    dataset = Dataset.from_dict({"system": systems, **targets})

    return dataset, target_info_dictionary
