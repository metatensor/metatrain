from typing import Dict, List, Union

import torch

from metatensor.models.utils.data import Dataset

from .dataset import DatasetInfo, TargetInfo


def get_targets_dict(
    datasets: List[Union[Dataset, torch.utils.data.Subset]], dataset_info: DatasetInfo
) -> Dict[str, TargetInfo]:
    """
    This is a helper function that extracts all the possible targets and their
    gradients from a list of datasets.

    :param datasets: A list of Datasets or Subsets.
    :param dataset_info: A DatasetInfo object containing further
        information about the dataset, namely the unit and quantity of the
        targets.

    :returns: A dictionary mapping target names to ``TargetInfo`` objects.

    :raises ValueError: If the ``DatasetInfo`` object does not contain any of
        the expected targets.
    """

    targets_dict = {}
    for dataset in datasets:
        targets = next(iter(dataset))
        targets.pop("system")  # system not needed

        # targets is now a dictionary of TensorMaps
        for target_name, target_tmap in targets.items():
            if target_name not in dataset_info.targets.keys():
                raise ValueError(
                    f"Target {target_name} not found in the targets 
                    specified in dataset_info."
                )
            if target_name not in targets_dict:
                targets_dict[target_name] = TargetInfo(
                    quantity=dataset_info.targets[target_name].quantity,
                    unit=dataset_info.targets[target_name].unit,
                    gradients=target_tmap.block(0).gradients_list(),
                )

    return targets_dict
