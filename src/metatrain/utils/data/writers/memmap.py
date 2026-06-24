from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from metatensor.torch import TensorMap
from metatomic.torch import ModelCapabilities, System

from .writers import Writer


class MemMapWriter(Writer):
    """
    Write predictions to numpy memory-mapped files, one ``.npy`` file per target.

    Values from every batch are buffered in memory during :meth:`write` calls.
    :meth:`finish` concatenates them and writes each target (and any gradient
    block) to a separate ``.npy`` file under the same directory as *filename*,
    using *filename*'s stem as a prefix::

        {stem}_{target_name}.npy
        {stem}_{target_name}_{gradient_name}_gradients.npy

    The resulting files can be memory-mapped with::

        arr = numpy.load(path, mmap_mode='r')

    Only single-block :class:`metatensor.torch.TensorMap` outputs are supported.
    For multi-block targets use the ``.mts`` format instead.

    :param filename: Base path for output files (e.g.
        ``checkpoint/final_evaluation/train_predictions.npy``).
    :param capabilities: Unused; kept to match the base-class signature.
    :param append: Unused; kept to match the base-class signature.
    """

    def __init__(
        self,
        filename: Union[str, Path],
        capabilities: Optional[ModelCapabilities] = None,
        append: Optional[bool] = None,
    ) -> None:
        super().__init__(filename, capabilities, append)
        self._buffers: Dict[str, List[np.ndarray]] = {}

    def write(self, systems: List[System], predictions: Dict[str, TensorMap]) -> None:
        """Buffer batch predictions in memory.

        :param systems: Unused; kept to match the base-class signature.
        :param predictions: Per-target TensorMaps produced by the model.
        :raises ValueError: If any TensorMap contains more than one block.
        """
        for target_name, tensor_map in predictions.items():
            if len(tensor_map.keys) != 1:
                raise ValueError(
                    f"MemMapWriter only supports single-block TensorMaps, but "
                    f"target '{target_name}' has {len(tensor_map.keys)} blocks. "
                    "Use the '.mts' format for multi-block targets."
                )
            block = tensor_map.block()
            values = block.values.detach().cpu().to(torch.float64).numpy()
            self._buffers.setdefault(target_name, []).append(values)

            for gradient_name, gradient_block in block.gradients():
                key = f"{target_name}_{gradient_name}_gradients"
                grad_values = (
                    gradient_block.values.detach().cpu().to(torch.float64).numpy()
                )
                self._buffers.setdefault(key, []).append(grad_values)

    def finish(self) -> None:
        """Concatenate buffered arrays and write one ``.npy`` file per target."""
        if not self._buffers:
            return

        stem = Path(self.filename).stem
        parent = Path(self.filename).parent
        parent.mkdir(parents=True, exist_ok=True)

        for name, arrays in self._buffers.items():
            concatenated = np.concatenate(arrays, axis=0)
            out_path = parent / f"{stem}_{name}.npy"
            fp = np.memmap(
                out_path,
                dtype=concatenated.dtype,
                mode="w+",
                shape=concatenated.shape,
            )
            fp[:] = concatenated
            del fp
