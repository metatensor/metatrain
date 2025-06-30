from pathlib import Path
from typing import Dict, List, Optional, Union

import ase
import ase.io
import torch
from metatensor.torch import TensorMap
from metatomic.torch import ModelCapabilities, System

from metatrain.utils.external_naming import to_external_name

from .writers import Writer


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
