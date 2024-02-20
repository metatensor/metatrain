from typing import List, Union

import torch
from metatensor.learn.data import Dataset


def get_outputs_dict(datasets: List[Union[Dataset, torch.utils.data.Subset]]):
    """
    This is a helper function that extracts all the possible outputs and their gradients
    from a list of datasets.

    :param datasets: A list of Datasets or Subsets.

    :returns: A dictionary mapping output names to a list of "values" (always)
        and possible gradients.
    """

    outputs_dict = {}
    for dataset in datasets:
        targets = next(iter(dataset))._asdict()
        targets.pop("structure")  # structure not needed

        # targets is now a dictionary of TensorMaps
        for target_name, target_tmap in targets.items():
            if target_name not in outputs_dict:
                outputs_dict[target_name] = [
                    "values"
                ] + target_tmap.block().gradients_list()

    return outputs_dict
