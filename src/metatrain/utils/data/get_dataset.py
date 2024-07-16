from pathlib import Path
from typing import Tuple

from omegaconf import DictConfig

from .dataset import Dataset, DiskDataset, TargetInfo, TargetInfoDict
from .readers import read_systems, read_targets


def get_dataset(
    options: DictConfig,
) -> Tuple[Dataset, TargetInfoDict]:
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

    if Path(options["systems"]["read_from"]).suffix == "":
        # metatensor disk dataset
        dataset = DiskDataset(options["systems"]["read_from"])
        target_info_dictionary = TargetInfoDict()
        # TODO: generalize this
        if len(options["targets"] != 1):
            raise ValueError("DiskDataset currently only supports a single target.")
        if "energy" not in options["targets"]:
            raise ValueError("DiskDataset currently only supports energy as a target.")
        target_info_dictionary["energy"] = TargetInfo(
            quantity=options["targets"]["energy"]["quantity"],
            unit=options["targets"]["energy"]["unit"],
            per_atom=False,
            gradients={"positions"},
        )
    else:
        systems = read_systems(
            filename=options["systems"]["read_from"],
            reader=options["systems"]["reader"],
        )
        targets, target_info_dictionary = read_targets(conf=options["targets"])
        dataset = Dataset({"system": systems, **targets})

    return dataset, target_info_dictionary
