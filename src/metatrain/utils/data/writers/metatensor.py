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
        self._systems: List[System] = []
        self._preds: List[Dict[str, TensorMap]] = []

    def write(self, systems: List[System], predictions: Dict[str, TensorMap]) -> None:
        """
        Accumulate systems and predictions to write them all at once in ``finish``.

        :param systems: List of systems to write.
        :param predictions: Dictionary of TensorMaps with predictions for the systems.
        """
        # just accumulate
        self._systems.extend(systems)
        self._preds.append(predictions)

    def finish(self) -> None:
        """
        Write all accumulated systems and predictions to Metatensor files.
        """
        # concatenate per-sample TensorMaps into full ones
        predictions = _concatenate_tensormaps(self._preds)
        # write out .mts files (writes one file per target)
        filename_base = Path(self.filename).stem
        for prediction_name, prediction_tmap in predictions.items():
            mts.save(
                filename_base + "_" + prediction_name + ".mts",
                prediction_tmap.to("cpu").to(torch.float64),
            )


def _concatenate_tensormaps(
    tensormap_dict_list: List[Dict[str, TensorMap]],
) -> Dict[str, TensorMap]:
    # Concatenating TensorMaps is tricky, because the model does not know the
    # "number" of the system it is predicting. For example, if a model predicts
    # 3 batches of 4 atoms each, the system labels will be [0, 1, 2, 3],
    # [0, 1, 2, 3], [0, 1, 2, 3] for the three batches, respectively. Due
    # to this, the join operation would not achieve the desired result
    # ([0, 1, 2, ..., 11, 12]). Here, we fix this by renaming the system labels.

    system_counter = 0
    n_systems = 0
    tensormaps_shifted_systems = []
    for tensormap_dict in tensormap_dict_list:
        tensormap_dict_shifted = {}
        for name, tensormap in tensormap_dict.items():
            new_keys = []
            new_blocks = []
            for key, block in tensormap.items():
                new_key = key
                where_system = block.samples.names.index("system")
                n_systems = torch.max(block.samples.column("system")) + 1
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
                    new_block.add_gradient(
                        gradient_name,
                        gradient_block,
                    )
                new_keys.append(new_key)
                new_blocks.append(new_block)
            tensormap_dict_shifted[name] = TensorMap(
                keys=Labels(
                    names=tensormap.keys.names,
                    values=torch.stack([new_key.values for new_key in new_keys]),
                ),
                blocks=new_blocks,
            )
        tensormaps_shifted_systems.append(tensormap_dict_shifted)
        system_counter += n_systems

    return {
        target: mts.join(
            [pred[target] for pred in tensormaps_shifted_systems],
            axis="samples",
            remove_tensor_name=True,
        )
        for target in tensormaps_shifted_systems[0].keys()
    }
