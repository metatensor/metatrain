from typing import Dict, List, Union

import torch
import warnings
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import System, ModelEvaluationOptions, register_autograd_neighbors

from .loss import TensorMapDictLoss
from .output_gradient import compute_gradient
from .export import is_exported
from .neighbor_list import calculate_neighbor_lists


# Ignore metatensor-torch warning due to the fact that positions/cell
# already require grad when registering the NL
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="neighbors",
)  # TODO: this is not filtering out the warning for some reason


def compute_model_loss(
    loss: TensorMapDictLoss,
    model: Union[torch.nn.Module, torch.jit._script.RecursiveScriptModule],
    systems: List[System],
    targets: Dict[str, TensorMap],
):
    """
    Compute the loss of a model on a set of targets.

    :param loss: The loss function to use.
    :param model: The model to use. This can either be a model in training
        (``torch.nn.Module``) or an exported model
        (``torch.jit._script.RecursiveScriptModule``).
    :param systems: The systems to use.
    :param targets: The targets to use.

    :returns: The loss as a scalar `torch.Tensor`.
    """
    # Assert that all targets are within the model's capabilities:
    if not set(targets.keys()).issubset(_get_capabilities(model).outputs.keys()):
        raise ValueError("Not all targets are within the model's capabilities.")
    
    # calculate neighbor lists (only for an exported model)
    if is_exported(model):
        requested_neighbor_lists = model.requested_neighbors_lists()
        systems = [calculate_neighbor_lists(system, requested_neighbor_lists) for system in systems]

    # Infer model device, move systems and targets to the same device:
    device = next(model.parameters()).device
    systems = [system.to(device=device) for system in systems]
    targets = {key: target.to(device=device) for key, target in targets.items()}

    # Find if there are any energy targets that require gradients:
    energy_targets = []
    energy_targets_that_require_position_gradients = []
    energy_targets_that_require_strain_gradients = []
    for target_name in targets.keys():
        # Check if the target is an energy:
        if _get_capabilities(model).outputs[target_name].quantity == "energy":
            energy_targets.append(target_name)
            # Check if the energy requires gradients:
            if targets[target_name].block().has_gradient("positions"):
                energy_targets_that_require_position_gradients.append(target_name)
            if targets[target_name].block().has_gradient("strain"):
                energy_targets_that_require_strain_gradients.append(target_name)

    if len(energy_targets_that_require_strain_gradients) > 0:
        # TODO: raise an error if the systems do not have a cell
        # if not all([system.has_cell for system in systems]):
        #     raise ValueError("One or more systems does not have a cell.")
        strains = [
            torch.eye(
                3,
                requires_grad=True,
                dtype=system.cell.dtype,
                device=system.cell.device,
            )
            for system in systems
        ]
        # Create new "displaced" systems:
        new_systems = []
        for system, strain in zip(systems, strains):
            new_system = System(
                positions=system.positions @ strain,
                cell=system.cell @ strain,
                species=system.species,
            )
            for nl_options in system.known_neighbors_lists():
                nl = system.get_neighbors_list(nl_options)
                register_autograd_neighbors(
                    new_system,
                    TensorBlock(
                        values=nl.values.detach(),
                        samples=nl.samples,
                        components=nl.components,
                        properties=nl.properties,
                    ),
                    check_consistency=True,
                )
                new_system.add_neighbors_list(nl_options, nl)
            new_systems.append(new_system)
        systems = new_systems
    else:
        if len(energy_targets_that_require_position_gradients) > 0:
            # Set positions to require gradients:
            for system in systems:
                system.positions.requires_grad_(True)

    # Based on the keys of the targets, get the outputs of the model:
    model_outputs = _get_model_outputs(model, systems, targets.keys())
    
    # model(
    #     systems, {key: _get_capabilities(model).outputs[key] for key in targets.keys()}
    # )

    for energy_target in energy_targets:
        # If the energy target requires gradients, compute them:
        target_requires_pos_gradients = (
            energy_target in energy_targets_that_require_position_gradients
        )
        target_requires_disp_gradients = (
            energy_target in energy_targets_that_require_strain_gradients
        )
        if target_requires_pos_gradients and target_requires_disp_gradients:
            gradients = compute_gradient(
                model_outputs[energy_target].block().values,
                [system.positions for system in systems] + strains,
                is_training=True,
            )
            old_energy_tensor_map = model_outputs[energy_target]
            new_block = old_energy_tensor_map.block().copy()
            new_block.add_gradient(
                "positions", _position_gradients_to_block(gradients[: len(systems)])
            )
            new_block.add_gradient(
                "strain",
                _strain_gradients_to_block(gradients[len(systems) :]),
            )
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
                strains,
                is_training=True,
            )
            old_energy_tensor_map = model_outputs[energy_target]
            new_block = old_energy_tensor_map.block().copy()
            new_block.add_gradient("strain", _strain_gradients_to_block(gradients))
            new_energy_tensor_map = TensorMap(
                keys=old_energy_tensor_map.keys,
                blocks=[new_block],
            )
            model_outputs[energy_target] = new_energy_tensor_map
        else:
            pass

    # Compute and return the loss and associated info:
    return loss(model_outputs, targets)


def _position_gradients_to_block(gradients_list):
    """Convert a list of position gradients to a `TensorBlock`
    which can act as a gradient block to an energy block."""

    # `gradients` consists of a list of tensors where the second dimension is 3
    gradients = torch.concatenate(gradients_list, dim=0).unsqueeze(-1)
    # unsqueeze for the property dimension

    samples = Labels(
        names=["sample", "atom"],
        values=torch.stack(
            [
                torch.concatenate(
                    [
                        torch.tensor([i] * len(structure))
                        for i, structure in enumerate(gradients_list)
                    ]
                ),
                torch.concatenate(
                    [torch.arange(len(structure)) for structure in gradients_list]
                ),
            ],
            dim=1,
        ),
    )

    components = [
        Labels(
            names=["xyz"],
            values=torch.tensor([[0], [1], [2]]),
        )
    ]

    return TensorBlock(
        values=gradients,
        samples=samples.to(gradients.device),
        components=[c.to(gradients.device) for c in components],
        properties=Labels.single().to(gradients.device),
    )


def _strain_gradients_to_block(gradients_list):
    """Convert a list of strain gradients to a `TensorBlock`
    which can act as a gradient block to an energy block."""

    gradients = torch.stack(gradients_list, dim=0).unsqueeze(-1)
    # unsqueeze for the property dimension

    samples = Labels(
        names=["sample"], values=torch.arange(len(gradients_list)).unsqueeze(-1)
    )

    components = [
        Labels(
            names=["xyz_1"],
            values=torch.tensor([[0], [1], [2]]),
        ),
        Labels(
            names=["xyz_2"],
            values=torch.tensor([[0], [1], [2]]),
        ),
    ]

    return TensorBlock(
        values=gradients,
        samples=samples.to(gradients.device),
        components=[c.to(gradients.device) for c in components],
        properties=Labels.single().to(gradients.device),
    )


def _get_capabilities(model: Union[torch.nn.Module, torch.jit._script.RecursiveScriptModule]):
    if is_exported(model):
        return model.capabilities()
    else:
        return model.capabilities
    
def _get_model_outputs(
    model: Union[torch.nn.Module, torch.jit._script.RecursiveScriptModule],
    systems: List[System],
    targets: List[str],
) -> Dict[str, TensorMap]:
    if is_exported(model):
        # put together an EvaluationOptions object
        options = ModelEvaluationOptions(
            length_unit = "",  # this is only needed for unit conversions in MD engines
            outputs = {key: _get_capabilities(model).outputs[key] for key in targets}
        )
        # we check consistency here because this could be called from eval
        return model(systems, options, check_consistency=True)
    else:
        return model(systems, {key: _get_capabilities(model).outputs[key] for key in targets})
