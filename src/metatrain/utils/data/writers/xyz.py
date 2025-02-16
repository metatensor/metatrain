from typing import Dict, List

import ase
import ase.io
import metatensor.torch
import torch
from metatensor.torch import Labels, TensorMap
from metatensor.torch.atomistic import ModelCapabilities, System

from ...external_naming import to_external_name


def write_xyz(
    filename: str,
    systems: List[System],
    capabilities: ModelCapabilities,
    predictions: Dict[str, TensorMap],
) -> None:
    """An ase-based xyz file writer. Writes the systems and predictions to an xyz file.

    According to ASE practice, arrays which have a dimension corresponding
    to each atom are saved inside atoms.arrays, while any other arrays are
    saved inside atoms.info.

    :param filename: name of the file to save to.
    :param systems: structures to be written to the file.
    :param: capabilities: capabilities of the model.
    :param predictions: prediction values to be written to the file.
    """

    # we first split the predictions by structure
    predictions_by_structure: List[Dict[str, TensorMap]] = [{} for _ in systems]
    split_labels = [
        Labels(names=["system"], values=torch.tensor([[i_system]]))
        for i_system in range(len(systems))
    ]
    for target_name, target_tensor_map in predictions.items():
        # split this target by structure
        target_tensor_map = target_tensor_map.to("cpu")
        split_target = metatensor.torch.split(
            target_tensor_map, "samples", split_labels
        )
        for i_system, system_target in enumerate(split_target):
            # add the split target to the dict corresponding to the structure
            predictions_by_structure[i_system][target_name] = system_target

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
                # reshaping here is necessary because `arrays` only accepts 2D arrays
            else:
                # save inside info
                if block.values.numel() == 1:
                    info[target_name] = block.values.item()
                else:
                    info[target_name] = block.values.detach().cpu().numpy().squeeze(0)
                    # squeeze the sample dimension, which corresponds to the system

            for gradient_name, gradient_block in block.gradients():
                # here, we assume that gradients are always an array, and never a scalar
                internal_name = f"{target_name}_{gradient_name}_gradients"
                external_name = to_external_name(internal_name, capabilities.outputs)

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
                            "stresses cannot be written for systems with zero volume."
                        )
                    info[external_name_virial] = -strain_derivatives
                    info[external_name_stress] = strain_derivatives / cell_volume
                else:
                    info[external_name] = (
                        # squeeze the property dimension
                        gradient_block.values.detach().cpu().squeeze(-1).numpy()
                    )

        atoms = ase.Atoms(
            symbols=system.types, positions=system.positions.detach(), info=info
        )

        # assign cell and pbcs
        if torch.any(system.cell != 0):
            atoms.pbc = True
            atoms.cell = system.cell.detach().cpu().numpy()

        # assign arrays
        for array_name, array in arrays.items():
            atoms.arrays[array_name] = array

        frames.append(atoms)

    ase.io.write(filename, frames)
