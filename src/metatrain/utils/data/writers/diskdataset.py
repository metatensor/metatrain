import zipfile
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import metatomic.torch as mta
import numpy as np
import torch
from metatensor.torch import TensorMap
from metatomic.torch import ModelCapabilities, System

from .writers import Writer, _split_tensormaps


class DiskDatasetWriter(Writer):
    """
    Write systems and predictions to a zip file, each system in a separate folder inside
    the zip.

    :param path: Path to the output zip file.
    :param capabilities: Model capabilities.
    :param append: If True, open the zip file in append mode.
    """

    def __init__(
        self,
        path: Union[str, Path],
        capabilities: Optional[
            ModelCapabilities
        ] = None,  # unused, but matches base signature
        append: Optional[bool] = False,  # if True, open zip in append mode
    ):
        super().__init__(filename=path, capabilities=capabilities, append=append)
        mode: Literal["w", "a"] = "a" if append else "w"
        self.zip_file = zipfile.ZipFile(path, mode)
        self.index = 0

    def write(self, systems: List[System], predictions: Dict[str, TensorMap]) -> None:
        """
        Write a single (system, predictions) into the zip under
        a new folder "<index>/".

        :param systems: List of systems to write.
        :param predictions: Dictionary of TensorMaps with predictions for the systems.
        """

        if len(systems) == 1:
            # Avoid reindexing samples
            split_predictions = [predictions]
        else:
            split_predictions = _split_tensormaps(
                systems, predictions, istart_system=self.index
            )

        for system, preds in zip(systems, split_predictions, strict=True):
            # system
            with self.zip_file.open(f"{self.index}/system.mta", "w") as f:
                mta.save(f, system.to("cpu").to(torch.float64))

            # each target
            for target_name, tensor_map in preds.items():
                with self.zip_file.open(f"{self.index}/{target_name}.mts", "w") as f:
                    buf = tensor_map.to("cpu").to(torch.float64)
                    # metatensor.torch.save_buffer returns a torch.Tensor buffer
                    buffer = buf.save_buffer()
                    np.save(f, buffer.numpy())

            self.index += 1

    def finish(self) -> None:
        """
        Close the zip file.
        """
        self.zip_file.close()
