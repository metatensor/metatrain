from typing import List, Optional

from pathlib import Path
from metatensor.torch import TensorMap
from metatensor.torch.atomistic import System

from .xyz import write_xyz


PREDICTIONS_WRITERS = {".xyz": write_xyz}
""":py:class:`dict`: dictionary mapping file suffixes to a prediction writers"""


def write_predictions(
    filename: str,
    predictions: TensorMap,
    structures: List[System],
    fileformat: Optional[str] = None,
) -> None:
    """Writes predictions to a file.

    For certain file suffixes also the structures will be written (i.e ``xyz``).

    :param filename: name of the file to write
    :param predictions: :py:class:`metatensor.torch.TensorMap` containing the
        predictions that should be written
    :param structures: list of structures that for some writers will also be written
    :param fileformat: format of the target value file. If :py:obj:`None` the format is
        determined from the suffix.
    """
    if fileformat is None:
        fileformat = Path(filename).suffix

    try:
        writer = PREDICTIONS_WRITERS[fileformat]
    except KeyError:
        raise ValueError(f"fileformat '{fileformat}' is not supported")

    return writer(filename, predictions, structures)
