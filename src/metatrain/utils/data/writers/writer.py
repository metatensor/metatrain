# writer.py
import zipfile
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import ase
import ase.io
import metatensor.torch
import metatomic.torch as mta
import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelCapabilities, System

from metatrain.utils.external_naming import to_external_name


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


class ASEWriter(Writer):
    """Streams out one frame at a time via write_xyz."""

    def __init__(
        self,
        filename: Union[str, Path],
        capabilities: Optional[
            ModelCapabilities
        ] = None,  # unused, but matches base signature
        append: Optional[bool] = False,  # unused, but matches base signature
    ):
        super().__init__(filename, capabilities, append)
        self._first = True

        self._systems: List[System] = []
        self._preds: List[Dict[str, TensorMap]] = []

    def write(self, system: System, predictions: Dict[str, TensorMap]):
        # just accumulate
        self._systems.append(system)
        self._preds.append(predictions)

    def finish(self):
        if not self._systems:
            return

        systems = self._systems
        predictions_by_structure = self._preds

        frames = []
        for system, system_predictions in zip(systems, predictions_by_structure):
            info = {}
            arrays = {}
            for target_name, target_map in system_predictions.items():
                if len(target_map.keys) != 1:
                    raise ValueError(
                        "Only single-block `TensorMap`s can be "
                        "written to xyz files for the moment."
                    )
                block = target_map.block()
                if "atom" in block.samples.names:
                    # save inside arrays
                    values = block.values.detach().cpu().numpy()
                    arrays[target_name] = values.reshape(values.shape[0], -1)
                    # reshaping reshaping because `arrays` only accepts 2D arrays
                else:
                    # save inside info
                    if block.values.numel() == 1:
                        info[target_name] = block.values.item()
                    else:
                        info[target_name] = (
                            block.values.detach().cpu().numpy().squeeze(0)
                        )
                        # squeeze the sample dimension, which corresponds to the system

                for gradient_name, gradient_block in block.gradients():
                    # we assume that gradients are always an array, never a scalar
                    internal_name = f"{target_name}_{gradient_name}_gradients"
                    external_name = to_external_name(
                        internal_name, self.capabilities.outputs
                    )

                    if "forces" in external_name:
                        arrays[external_name] = (
                            # squeeze the property dimension
                            -gradient_block.values.detach().cpu().squeeze(-1).numpy()
                        )
                    elif "virial" in external_name:
                        # in this case, we write both the virial and the stress
                        external_name_virial = external_name
                        external_name_stress = external_name.replace("virial", "stress")
                        strain_derivatives = (
                            # squeeze the property dimension
                            gradient_block.values.detach().cpu().squeeze(-1).numpy()
                        )
                        if not torch.any(system.cell != 0):
                            raise ValueError(
                                "stresses cannot be written for non-periodic systems."
                            )
                        cell_volume = torch.det(system.cell).item()
                        if cell_volume == 0:
                            raise ValueError(
                                (
                                    "stresses cannot be written for "
                                    "systems with zero volume."
                                )
                            )
                        info[external_name_virial] = -strain_derivatives
                        info[external_name_stress] = strain_derivatives / cell_volume
                    else:
                        info[external_name] = (
                            # squeeze the property dimension
                            gradient_block.values.detach().cpu().squeeze(-1).numpy()
                        )

            atoms = ase.Atoms(
                symbols=system.types.numpy(),
                positions=system.positions.detach().numpy(),
                info=info,
            )

            # assign cell and pbcs
            if torch.any(system.cell != 0):
                atoms.pbc = True
                atoms.cell = system.cell.detach().cpu().numpy()

            # assign arrays
            for array_name, array in arrays.items():
                atoms.arrays[array_name] = array

            frames.append(atoms)

        ase.io.write(self.filename, frames)


class MetatensorWriter(Writer):
    """Buffers all samples in memory, then emits full .mts files at finish()."""

    def __init__(
        self,
        filename: Union[str, Path],
        capabilities: Optional[ModelCapabilities] = None,
        append: Optional[bool] = False,  # unused, but matches base signature
    ):
        super().__init__(filename, capabilities, append)
        self._systems: List[System] = []
        self._preds: List[Dict[str, TensorMap]] = []

    def write(self, system: System, predictions: Dict[str, TensorMap]):
        # just accumulate
        self._systems.append(system)
        self._preds.append(predictions)

    def finish(self):
        # concatenate per-sample TensorMaps into full ones
        predictions = _concatenate_tensormaps(self._preds)
        # write out .mts files (writes one file per target)
        filename_base = Path(self.filename).stem
        for prediction_name, prediction_tmap in predictions.items():
            metatensor.torch.save(
                filename_base + "_" + prediction_name + ".mts",
                prediction_tmap.to("cpu").to(torch.float64),
            )


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
                new_samples_values = block.samples.values
                new_samples_values[:, where_system] += system_counter
                new_block = TensorBlock(
                    values=block.values,
                    samples=Labels(block.samples.names, values=new_samples_values),
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
        target: metatensor.torch.join(
            [pred[target] for pred in tensormaps_shifted_systems],
            axis="samples",
            remove_tensor_name=True,
        )
        for target in tensormaps_shifted_systems[0].keys()
    }
