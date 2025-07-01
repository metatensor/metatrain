# writer.py
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import metatensor.torch
import metatomic.torch as mta
import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelCapabilities, System


class Writer(ABC):
    def __init__(
        self,
        filename: Union[str, Path],
        capabilities: Optional[ModelCapabilities] = None,
        append: Optional[bool] = None,
    ):
        self.filename = filename
        self.capabilities = capabilities
        self.append = append

    @abstractmethod
    def write(self, systems: List[System], predictions: Dict[str, TensorMap]):
        """Write a single system + its predictions."""
        ...

    @abstractmethod
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

    def write(self, systems: List[System], predictions: Dict[str, TensorMap]):
        """
        Write a single (system, predictions) into the zip under
        a new folder "<index>/".
        """

        if len(systems) == 1:
            # Avoid reindexing samples
            split_predictions = [predictions]
        else:
            split_predictions = _split_tensormaps(
                systems, predictions, istart_system=self.index
            )

        for system, preds in zip(systems, split_predictions):
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

    def finish(self):
        self.zip_file.close()


def _split_tensormaps(
    systems: List[System],
    batch_predictions: Dict[str, TensorMap],
    istart_system: Optional[int] = 0,
) -> List[Dict[str, TensorMap]]:
    """
    Split a TensorMap into multiple TensorMaps, one for each key.
    """

    device = next(iter(batch_predictions.values()))[0].values.device

    split_selection = [
        Labels("system", torch.tensor([[i]], device=device))
        for i in range(len(systems))
    ]
    batch_predictions_split = {
        key: metatensor.torch.split(tensormap, "samples", split_selection)
        for key, tensormap in batch_predictions.items()
    }

    out_tensormaps: List[Dict[str, TensorMap]] = []
    for i in range(len(systems)):
        # build a per-sample dict
        tensormaps: Dict[str, TensorMap] = {}
        for k in batch_predictions_split.keys():
            new_blocks: List[TensorBlock] = []
            for block in batch_predictions_split[k][i]:
                new_block = TensorBlock(
                    samples=Labels(
                        block.samples.names,
                        block.samples.values
                        + istart_system
                        * torch.eye(
                            block.samples.values.size(-1),
                            device=block.samples.values.device,
                            dtype=block.samples.values.dtype,
                        )[0],
                    ),
                    components=block.components,
                    properties=block.properties,
                    values=block.values,
                )
                for gradient_name, gradient_block in block.gradients():
                    new_block.add_gradient(
                        gradient_name,
                        TensorBlock(
                            samples=Labels(
                                gradient_block.samples.names,
                                gradient_block.samples.values
                                + istart_system
                                * torch.eye(
                                    gradient_block.samples.values.size(-1),
                                    device=gradient_block.samples.values.device,
                                    dtype=gradient_block.samples.values.dtype,
                                )[0],
                            ),
                            components=gradient_block.components,
                            properties=gradient_block.properties,
                            values=gradient_block.values,
                        ),
                    )
                new_blocks.append(new_block)
            tensormaps[k] = TensorMap(
                keys=batch_predictions_split[k][i].keys,
                blocks=new_blocks,
            )

        out_tensormaps.append(tensormaps)

    return out_tensormaps
