from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import metatensor.torch
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
        """Write a single system and its predictions."""
        ...

    @abstractmethod
    def finish(self):
        """Called after all writes. Optional to override."""
        ...


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
