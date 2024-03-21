from typing import Dict, List

import ase
import ase.io
import metatensor.torch
import numpy as np
import torch
from metatensor.torch import Labels, TensorMap
from metatensor.torch.atomistic import ModelCapabilities, System


def write_xyz(
    filename: str,
    systems: List[System],
    capabilities: ModelCapabilities,
    predictions: Dict[str, TensorMap],
) -> None:
    """An ase-based xyz file writer. Writes the systems and predictions to an xyz file.

    :param filename: name of the file to read.
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
        split_target = metatensor.torch.split(
            target_tensor_map, "samples", split_labels
        )
        for i_system, system_target in enumerate(split_target):
            # add the split target to the dict corresponding to the structure
            predictions_by_structure[i_system][target_name] = system_target

    only_one_energy = (
        np.sum(
            [
                capabilities.outputs[key].quantity == "energy"
                for key in predictions.keys()
            ]
        )
        == 1
    )

    frames = []
    for system, system_predictions in zip(systems, predictions_by_structure):
        info = {}
        arrays = {}
        for target_name, target_map in system_predictions.items():
            if len(target_map.keys.names) != 1:
                raise ValueError(
                    "Only single-block `TensorMap`s can be "
                    "written to xyz files for the moment."
                )
            block = target_map.block()
            if block.values.numel() == 1:  # this is a scalar
                info[target_name] = block.values.item()
            else:  # this is an array
                arrays[target_name] = block.values.detach().cpu().numpy()
            for gradient_name, gradient_block in block.gradients():
                # here, we assume that gradients are always an array, and never a scalar
                if capabilities.outputs[target_name].quantity == "energy":
                    if gradient_name == "positions":
                        if only_one_energy:
                            name_for_saving = "forces"
                        else:
                            name_for_saving = f"forces[{target_name}]"
                        arrays[name_for_saving] = (
                            # squeeze the property dimension
                            -gradient_block.values.detach()
                            .cpu()
                            .squeeze(-1)
                            .numpy()
                        )
                    elif gradient_name == "strain":
                        strain_derivatives = (
                            gradient_block.values.detach().cpu().numpy()
                        )
                        if not torch.any(system.cell != 0):
                            raise ValueError(
                                "stresses cannot be written for non-periodic systems."
                            )
                        cell_volume = torch.det(system.cell)
                        if cell_volume == 0:
                            raise ValueError(
                                "stresses cannot be written for "
                                "systems with zero volume."
                            )
                        if only_one_energy:
                            name_for_saving = "stress"
                        else:
                            name_for_saving = f"stress[{target_name}]"
                        arrays[name_for_saving] = strain_derivatives / cell_volume
                    else:
                        arrays[f"{target_name}_{gradient_name}_gradients"] = (
                            # squeeze the property dimension
                            gradient_block.values.detach()
                            .cpu()
                            .squeeze(-1)
                            .numpy()
                        )

        atoms = ase.Atoms(
            symbols=system.types, positions=system.positions.detach(), info=info
        )

        # assign cell and pbcs
        if torch.any(system.cell != 0):
            atoms.pbc = True
            atoms.cell = system.cell

        # assign arrays
        for array_name, array in arrays.items():
            atoms.arrays[array_name] = array

        frames.append(atoms)

    ase.io.write(filename, frames)
