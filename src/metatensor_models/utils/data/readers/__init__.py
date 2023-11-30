""""Readers for structures and target values."""

from typing import List, Dict, Optional

from pathlib import Path

from metatensor.torch import TensorMap

from .structures import STRUCTURE_READERS
from .targets import TARGET_READERS

from rascaline.torch.system import System


def read_structures(filename: str, fileformat: Optional[str] = None) -> List[System]:
    """Reads a structure information from file."""

    if fileformat is None:
        fileformat = Path(filename).suffix

    try:
        reader = STRUCTURE_READERS[fileformat]
    except KeyError:
        raise ValueError(f"fileformat '{fileformat}' is not supported")

    return reader(filename)


def read_targets(
    filename: str,
    target_value: str,
    fileformat: Optional[str] = None,
) -> Dict[str, TensorMap]:
    """Reads target information from file."""

    if fileformat is None:
        fileformat = Path(filename).suffix

    try:
        reader = TARGET_READERS[fileformat]
    except KeyError:
        raise ValueError(f"fileformat '{fileformat}' is not supported")

    return reader(filename, target_value)
