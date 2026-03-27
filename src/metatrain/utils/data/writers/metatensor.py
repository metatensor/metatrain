import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelCapabilities, System

from .writers import Writer


class MetatensorWriter(Writer):
    """
    Write systems and predictions to Metatensor files (.mts).

    Each ``write()`` call saves the batch predictions to temporary files on disk,
    avoiding unbounded memory growth. ``finish()`` loads the temp files, concatenates
    them with correct system label offsets, and writes the final output.

    :param filename: Base filename for the output files. Each target will be saved in a
        separate file with the target name appended.
    :param capabilities: Model capabilities.
    :param append: Whether to append to existing files, unused here but kept for
        compatibility with the base class.
    """

    def __init__(
        self,
        filename: Union[str, Path],
        capabilities: Optional[ModelCapabilities] = None,
        append: Optional[bool] = False,  # unused, but matches base signature
    ) -> None:
        super().__init__(filename, capabilities, append)
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._batch_idx = 0
        self._target_names: List[str] = []

    def write(self, systems: List[System], predictions: Dict[str, TensorMap]) -> None:
        """
        Save batch predictions to temporary files, freeing GPU/CPU memory immediately.

        :param systems: List of systems to write.
        :param predictions: Dictionary of TensorMaps with predictions for the systems.
        """
        if self._batch_idx == 0:
            self._target_names = list(predictions.keys())

        tmp_path = Path(self._tmp_dir.name)
        for target_name, tmap in predictions.items():
            fname = tmp_path / f"{self._batch_idx}_{target_name}.mts"
            mts.save(str(fname), tmap.to("cpu").to(torch.float64))

        self._batch_idx += 1

    def finish(self) -> None:
        """
        Load temp files, shift system labels, join, and write final .mts files.
        """
        if self._batch_idx == 0:
            return

        tmp_path = Path(self._tmp_dir.name)
        filename_base = Path(self.filename).with_suffix(".mts")

        for target_name in self._target_names:
            batch_tmaps = []
            for i in range(self._batch_idx):
                fname = tmp_path / f"{i}_{target_name}.mts"
                batch_tmaps.append(mts.load(str(fname)))

            merged = _concatenate_tensormaps_flat(batch_tmaps)
            filename = filename_base.with_stem(filename_base.stem + "_" + target_name)
            mts.save(filename, merged)

        self._tmp_dir.cleanup()


def _concatenate_tensormaps_flat(
    tensormap_list: List[TensorMap],
) -> TensorMap:
    """Concatenate a list of TensorMaps along the samples axis.

    Concatenating TensorMaps is tricky, because the model does not know the
    "number" of the system it is predicting. For example, if a model predicts
    3 batches of 4 atoms each, the system labels will be [0, 1, 2, 3],
    [0, 1, 2, 3], [0, 1, 2, 3] for the three batches, respectively. Due
    to this, a plain join would not produce the desired result
    ([0, 1, 2, ..., 11]). This function fixes that by shifting the system
    labels so they form a contiguous range across batches before joining.

    :param tensormap_list: List of TensorMaps to concatenate.
    :return: A single TensorMap with shifted system labels joined along samples.
    """

    system_counter = 0
    shifted = []
    for tensormap in tensormap_list:
        new_keys = []
        new_blocks = []
        n_systems = 0
        for key, block in tensormap.items():
            where_system = block.samples.names.index("system")
            n_systems = max(
                n_systems, int(torch.max(block.samples.column("system")).item()) + 1
            )
            new_samples_values = block.samples.values.clone()
            new_samples_values[:, where_system] += system_counter
            new_block = TensorBlock(
                values=block.values,
                samples=Labels(
                    block.samples.names,
                    values=new_samples_values,
                    assume_unique=True,
                ),
                components=block.components,
                properties=block.properties,
            )
            for gradient_name, gradient_block in block.gradients():
                new_block.add_gradient(gradient_name, gradient_block)
            new_keys.append(key)
            new_blocks.append(new_block)
        shifted.append(
            TensorMap(
                keys=Labels(
                    names=tensormap.keys.names,
                    values=torch.stack([k.values for k in new_keys]),
                ),
                blocks=new_blocks,
            )
        )
        system_counter += n_systems

    return mts.join(shifted, axis="samples")
