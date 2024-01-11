from typing import List, Dict, Optional, Union

from pathlib import Path

from metatensor.torch import TensorMap

from .structures import STRUCTURE_READERS
from .targets import TARGET_READERS

from metatensor.torch.atomistic import System


def read_structures(filename: str, fileformat: Optional[str] = None) -> List[System]:
    """Read structure informations from a file.

    :param filename: name of the file to read
    :param fileformat: format of the structure file. If :py:obj:`None` the format is
        determined from the suffix.
    :returns: list of structures
    """

    if fileformat is None:
        fileformat = Path(filename).suffix

    try:
        reader = STRUCTURE_READERS[fileformat]
    except KeyError:
        raise ValueError(f"fileformat '{fileformat}' is not supported")

    return reader(filename)


def read_targets(
    filename: str,
    target_values: Union[List[str], str],
    fileformat: Optional[str] = None,
) -> Dict[str, TensorMap]:
    """Read target informations from a file.

    :param filename: name of the file to read
    :param target_values: target values to be parsed from the file.
    :param fileformat: format of the target value file. If :py:obj:`None` the format is
        determined from the suffix.
    :returns: dictionary containing one key per ``target_value``.
    """

    if fileformat is None:
        fileformat = Path(filename).suffix

    try:
        reader = TARGET_READERS[fileformat]
    except KeyError:
        raise ValueError(f"fileformat '{fileformat}' is not supported")

    return reader(filename, target_values)
