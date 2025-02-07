from typing import Dict

from metatensor.torch import TensorMap, save
from typing import List
from metatensor.torch.atomistic import System, ModelCapabilities


# note that, although we don't use `systems` and `capabilities`, we need them to
# match the writer interface
def write_mts(
    filename: str,
    systems: List[System],
    capabilities: ModelCapabilities,
    predictions: Dict[str, TensorMap],
) -> None:
    """A metatensor-format prediction writer. Writes the predictions to `.mts` files.

    :param filename: name of the file to save to.
    :param systems: structures to be written to the file (not written by this writer).
    :param: capabilities: capabilities of the model (not used by this writer)
    :param predictions: prediction values to be written to the file.
    """

    for prediction_name, prediction_tmap in predictions:
        save(filename, prediction_tmap)
