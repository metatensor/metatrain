# writer.py
import zipfile
from pathlib import Path
from typing import Dict, Literal, Optional, Union

import metatomic.torch as mta
import numpy as np
import torch
from metatensor.torch import TensorMap
from metatomic.torch import ModelCapabilities, System


class Writer:
    def __init__(
        self,
        filename: Union[str, Path],
        capabilities: Optional[ModelCapabilities] = None,
        append: Optional[bool] = None,
    ):
        self.filename = filename
        self.capabilities = capabilities
        self.append = append

    def write(self, system: System, predictions: Dict[str, TensorMap]):
        """Write a single system + its predictions."""
        ...

    def finish(self):
        """Called after all writes. Optional to override."""
        ...


class DiskDatasetWriter(Writer):
    def __init__(
        self,
        path: Union[str, Path],
        capabilities: Optional[
            ModelCapabilities
        ] = None,  # unused, but matches base signature
        append: Optional[bool] = True,  # if True, open zip in append mode
    ):
        super().__init__(filename=path, capabilities=capabilities, append=append)
        mode: Literal["w", "a"] = "a" if append else "w"
        self.zip_file = zipfile.ZipFile(path, mode)
        self.index = 0

    def write(self, system: System, predictions: Dict[str, TensorMap]):
        """
        Write a single (system, predictions) into the zip under
        a new folder "<index>/".
        """
        # system
        with self.zip_file.open(f"{self.index}/system.mta", "w") as f:
            mta.save(f, system)

        # each target
        for target_name, tensor_map in predictions.items():
            with self.zip_file.open(f"{self.index}/{target_name}.mts", "w") as f:
                buf = tensor_map.to("cpu").to(torch.float64)
                # metatensor.torch.save_buffer returns a torch.Tensor buffer
                buffer = buf.save_buffer()
                np.save(f, buffer.numpy())

        self.index += 1

    def finish(self):
        self.zip_file.close()
