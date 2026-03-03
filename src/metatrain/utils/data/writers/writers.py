from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import metatensor.torch as mts
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
    def write(self, systems: List[System], predictions: Dict[str, TensorMap]) -> None:
        """
        Write a single system and its predictions.

        :param systems: List of systems to write.
        :param predictions: Dictionary of TensorMaps with predictions for the systems.
        """
        ...

    @abstractmethod
    def finish(self) -> None:
        """Called after all writes. Optional to override."""
        ...


def _split_tensormaps(
    systems: List[System],
    batch_predictions: Dict[str, TensorMap],
    istart_system: Optional[int] = 0,
) -> List[Dict[str, TensorMap]]:
    """Split batch predictions into per-system TensorMaps.

    After ``mts.split`` each sub-block contains a single system. The
    ``istart_system`` offset is added to the ``"system"`` column of block
    samples so that systems across successive batches receive contiguous
    global indices (needed by ``DiskDatasetWriter``).

    Gradient samples are left unchanged because their ``"sample"`` column
    is a **positional** reference into the parent block (always 0 after
    splitting to a single system). Offsetting it would make metatensor
    reject the gradient with an out-of-range sample index.

    :param systems: Systems in the current batch.
    :param batch_predictions: Predictions keyed by target name.
    :param istart_system: Global index of the first system in this batch.
    :return: List of per-system prediction dicts (length = len(systems)).
    """

    device = next(iter(batch_predictions.values()))[0].values.device

    split_selection = [
        Labels("system", torch.tensor([[i]], device=device))
        for i in range(len(systems))
    ]
    batch_predictions_split = {
        key: mts.split(tensormap, "samples", split_selection)
        for key, tensormap in batch_predictions.items()
    }

    out_tensormaps: List[Dict[str, TensorMap]] = []
    for i in range(len(systems)):
        tensormaps: Dict[str, TensorMap] = {}
        for k in batch_predictions_split.keys():
            new_blocks: List[TensorBlock] = []
            for block in batch_predictions_split[k][i]:
                # Offset "system" column (always column 0 in block samples)
                # so that systems across batches get unique global indices.
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
                        assume_unique=True,
                    ),
                    components=block.components,
                    properties=block.properties,
                    values=block.values,
                )
                # Gradient samples use a positional "sample" column that
                # indexes into the parent block (0 after splitting). It must
                # NOT be offset. Other columns ("atom", etc.) are
                # system-local and also stay unchanged.
                for gradient_name, gradient_block in block.gradients():
                    new_block.add_gradient(gradient_name, gradient_block)
                new_blocks.append(new_block)
            tensormaps[k] = TensorMap(
                keys=batch_predictions_split[k][i].keys,
                blocks=new_blocks,
            )

        out_tensormaps.append(tensormaps)

    return out_tensormaps
