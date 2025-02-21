from pathlib import Path
from typing import Dict, List

import torch
from metatensor.torch import TensorMap, save
from metatensor.torch.atomistic import ModelCapabilities, System


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

    filename_base = Path(filename).stem
    for prediction_name, prediction_tmap in predictions.items():
        save(
            filename_base + "_" + prediction_name + ".mts",
            prediction_tmap.to("cpu").to(torch.float64),
        )
