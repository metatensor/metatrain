import torch
from metatensor.torch.atomistic import System
from metatensor.torch import Labels, TensorBlock, TensorMap

from typing import Dict, List
from .loss import TensorMapDictLoss
from .output_gradient import compute_gradient


def compute_model_loss(
    loss: TensorMapDictLoss,
    model: torch.nn.Module,
    systems: List[System],
    targets: Dict[str, TensorMap],
):
    """
    Compute the loss of a model on a set of targets.

    This function assumes that the model returns a dictionary of
    TensorMaps, with the same keys as the targets.
    """
    # Assert that all targets are within the model's capabilities:
    if not set(targets.keys()).issubset(model.capabilities.outputs.keys()):
        raise ValueError("Not all targets are within the model's capabilities.")

    # Find if there are any energy targets that require gradients:
    energy_targets = []
    energy_targets_that_require_position_gradients = []
    energy_targets_that_require_displacement_gradients = []
    for target_name in targets.keys():
        # Check if the target is an energy:
        if model.capabilities.outputs[target_name].quantity == "energy":
            energy_targets.append(target_name)
            # Check if the energy requires gradients:
            if targets[target_name].has_gradients("positions"):
                energy_targets_that_require_position_gradients.append(target_name)
            if targets[target_name].has_gradients("displacements"):
                energy_targets_that_require_displacement_gradients.append(target_name)
                
    if len(energy_targets_that_require_displacement_gradients) > 0:
        # TODO: raise an error if the systems do not have a cell
        # if not all([system.has_cell for system in systems]):
        #     raise ValueError("One or more systems does not have a cell.")
        displacements = [torch.eye(3, requires_grad=True, dtype=system.dtype, device=system.device) for system in systems]
        # Create new "displaced" systems:
        systems = [
            System(
                positions=system.positions @ displacement,
                cell=system.cell @ displacement,
                species=system.species,
            )
            for system, displacement in zip(systems, displacements)
        ]
    else:
        if len(energy_targets_that_require_position_gradients) > 0:
            # Set positions to require gradients:
            for system in systems:
                system.positions.requires_grad_(True)

    # Based on the keys of the targets, get the outputs of the model:
    model_outputs = model(systems, targets.keys())

    for energy_target in energy_targets:
        # If the energy target requires gradients, compute them:
        target_requires_pos_gradients = energy_target in energy_targets_that_require_position_gradients
        target_requires_disp_gradients = energy_target in energy_targets_that_require_displacement_gradients
        if target_requires_pos_gradients and target_requires_disp_gradients:
            gradients = compute_gradient(
                model_outputs[energy_target].block().values,
                [system.positions for system in systems] + displacements,
                is_training=True,
            )
            old_energy_tensor_map = model_outputs[energy_target]
            new_block = old_energy_tensor_map.block().copy()
            new_block.add_gradient("positions", _position_gradients_to_block(gradients[:len(systems)]))
            new_block.add_gradient("displacements", _displacement_gradients_to_block(gradients[len(systems):]))
            new_energy_tensor_map = TensorMap(
                keys=old_energy_tensor_map.keys,
                blocks=[new_block],
            )
            model_outputs[energy_target] = new_energy_tensor_map
        elif target_requires_pos_gradients:
            gradients = compute_gradient(
                model_outputs[energy_target].block().values,
                [system.positions for system in systems],
                is_training=True,
            )
            old_energy_tensor_map = model_outputs[energy_target]
            new_block = old_energy_tensor_map.block().copy()
            new_block.add_gradient("positions", _position_gradients_to_block(gradients))
            new_energy_tensor_map = TensorMap(
                keys=old_energy_tensor_map.keys,
                blocks=[new_block],
            )
            model_outputs[energy_target] = new_energy_tensor_map
        elif target_requires_disp_gradients:
            gradients = compute_gradient(
                model_outputs[energy_target].block().values,
                displacements,
                is_training=True,
            )
            old_energy_tensor_map = model_outputs[energy_target]
            new_block = old_energy_tensor_map.block().copy()
            new_block.add_gradient("displacements", _displacement_gradients_to_block(gradients))
            new_energy_tensor_map = TensorMap(
                keys=old_energy_tensor_map.keys,
                blocks=[new_block],
            )
            model_outputs[energy_target] = new_energy_tensor_map
        else:
            pass

    # Compute the loss:
    return loss(model_outputs, targets)


def _position_gradients_to_block(gradients_list):
    """Convert a list of position gradients to a `TensorBlock` 
    which can act as a gradient block to an energy block."""

    # `gradients` consists of a list of tensors where the second dimension is 3
    gradients = torch.stack(gradients_list, dim=0).unsqueeze(-1)
    # unsqueeze for the property dimension

    samples = Labels(
        names=["sample", "atom"],
        values=torch.stack([
            torch.concatenate([torch.tensor([i]*len(structure)) for i, structure in enumerate(gradients_list)]),
            torch.concatenate([torch.arange(len(structure)) for structure in gradients_list]),
        ], dim=1),
    )

    components = [
        Labels(
            names=["coordinate"],
            values=torch.tensor([[0], [1], [2]]),
        )
    ]

    return TensorBlock(
        values=gradients,
        samples=samples,
        components=components,
        properties=Labels.single(),
    )


def _displacement_gradients_to_block(gradients_list):
    """Convert a list of displacement gradients to a `TensorBlock` 
    which can act as a gradient block to an energy block."""

    """Convert a list of position gradients to a `TensorBlock` 
    which can act as a gradient block to an energy block."""

    # `gradients` consists of a list of tensors where the second dimension is 3
    gradients = torch.stack(gradients_list, dim=0).unsqueeze(-1)
    # unsqueeze for the property dimension

    samples = Labels(
        names=["sample"],
        values=torch.arange(len(gradients_list)).unsqueeze(-1)
    )

    # TODO: check if this makes physical sense
    components = [
        Labels(
            names=["cell vector"],
            values=torch.tensor([[0], [1], [2]]),
        ),
        Labels(
            names=["coordinate"],
            values=torch.tensor([[0], [1], [2]]),
        )
    ]

    return TensorBlock(
        values=gradients,
        samples=samples,
        components=components,
        properties=Labels.single(),
    )
