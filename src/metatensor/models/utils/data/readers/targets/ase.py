from typing import Dict, List, Union

import ase.io
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


def read_ase(
    filename: str,
    target_values: Union[List[str], str],
) -> Dict[str, TensorMap]:
    """Store target informations with ase in a :class:`metatensor.TensorMap`.

    :param filename: name of the file to read
    :param target_values: target values to be parsed from the file.

    :returns:
        TensorMap containing the given information
    """

    if type(target_values) is str:
        target_values = [target_values]

    frames = ase.io.read(filename, ":")

    target_dictionary = {}
    for target_value in target_values:
        values = [f.info[target_value] for f in frames]

        n_structures = len(values)

        block = TensorBlock(
            values=torch.tensor(values).reshape(-1, 1),
            samples=Labels(["structure"], torch.arange(n_structures).reshape(-1, 1)),
            components=[],
            properties=Labels.single(),
        )

        target_dictionary[target_value] = TensorMap(
            keys=Labels(["lambda", "sigma"], torch.tensor([(0, 1)])), blocks=[block]
        )

    return target_dictionary
