from pathlib import Path
from typing import List, Optional

from metatensor.torch import TensorMap
from metatensor.torch.atomistic import ModelCapabilities, System

from .metatensor import write_mts
from .xyz import write_xyz


PREDICTIONS_WRITERS = {".xyz": write_xyz, ".mts": write_mts}
""":py:class:`dict`: dictionary mapping file suffixes to a prediction writers"""


def write_predictions(
    filename: str,
    systems: List[System],
    capabilities: ModelCapabilities,
    predictions: TensorMap,
    fileformat: Optional[str] = None,
) -> None:
    """Writes predictions to a file.

    For certain file suffixes, the systems will also be written (i.e ``xyz``).

    The capabilities of the model are used to infer the type (physical quantity) of
    the predictions. In this way, for example, position gradients of energies can be
    saved as forces.

    For the moment, strain gradients of the energy are saved as stresses
    (and not as virials).

    :param filename: name of the file to write
    :param systems: list of systems that for some writers will also be written
    :param capabilities: capabilities of the model
    :param predictions: :py:class:`metatensor.torch.TensorMap` containing the
        predictions that should be written
    :param fileformat: format of the target value file. If :py:obj:`None` the format is
        determined from the file extension.
    """
    if fileformat is None:
        fileformat = Path(filename).suffix

    try:
        writer = PREDICTIONS_WRITERS[fileformat]
    except KeyError:
        raise ValueError(f"fileformat '{fileformat}' is not supported")

    return writer(filename, systems, capabilities, predictions)
