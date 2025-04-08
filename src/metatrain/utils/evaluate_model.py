import warnings
from typing import Dict, List, Union

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelEvaluationOptions,
    ModelOutput,
    System,
    is_atomistic_model,
)

from .data import TargetInfo
from .output_gradient import compute_gradient


def evaluate_model(
    model: Union[
        torch.nn.Module,
        MetatensorAtomisticModel,
        torch.jit._script.RecursiveScriptModule,
    ],
    systems: List[System],
    targets: Dict[str, TargetInfo],
    is_training: bool,
    check_consistency: bool = False,
) -> Dict[str, TensorMap]:
    """
    Evaluate the model (in training or exported) on a set of requested targets.

    :param model: The model to use. This can either be a model in training
        (``torch.nn.Module``) or an exported model
        (``torch.jit._script.RecursiveScriptModule``).
    :param systems: The systems to use.
    :param targets: The names of the targets to evaluate (keys), along with
        their associated gradients (values).
    :param is_training: Whether the model is being computed during training.

    :returns: The predictions of the model for the requested targets.
    """

    model_outputs = _get_outputs(model)
    # Assert that all targets are within the model's capabilities:
    if not set(targets.keys()).issubset(model_outputs.keys()):
        raise ValueError("Not all targets are within the model's capabilities.")

    # Find if there are any energy targets that require gradients:
    energy_targets = []
    energy_targets_that_require_position_gradients = []
    energy_targets_that_require_strain_gradients = []
    for target_name in targets.keys():
        # Check if the target is an energy:
        if model_outputs[target_name].quantity == "energy":
            energy_targets.append(target_name)
            # Check if the energy requires gradients:
            if "positions" in targets[target_name].gradients:
                energy_targets_that_require_position_gradients.append(target_name)
            if "strain" in targets[target_name].gradients:
                energy_targets_that_require_strain_gradients.append(target_name)

    new_systems = []
    strains = []
    for system in systems:
        with warnings.catch_warnings():
            # this seems to be the only way to filter out the torch-scripted warnings
            # about neighbors (which are not relevant here), regex fails
            warnings.simplefilter("ignore")
            new_system, strain = _prepare_system(
                system,
                positions_grad=len(energy_targets_that_require_position_gradients) > 0,
                strain_grad=len(energy_targets_that_require_strain_gradients) > 0,
                check_consistency=check_consistency,
            )
        new_systems.append(new_system)
        strains.append(strain)
    systems = new_systems

    # Based on the keys of the targets, get the outputs of the model:
    model_outputs = _get_model_outputs(model, systems, targets, check_consistency)

    for energy_target in energy_targets:
        # If the energy target requires gradients, compute them:
        target_requires_pos_gradients = (
            energy_target in energy_targets_that_require_position_gradients
        )
        target_requires_strain_gradients = (
            energy_target in energy_targets_that_require_strain_gradients
        )
        if target_requires_pos_gradients and target_requires_strain_gradients:
            gradients = compute_gradient(
                model_outputs[energy_target].block().values,
                [system.positions for system in systems] + strains,
                is_training=is_training,
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
                is_training=is_training,
            )
            old_energy_tensor_map = model_outputs[energy_target]
            new_block = old_energy_tensor_map.block().copy()
            new_block.add_gradient("positions", _position_gradients_to_block(gradients))
            new_energy_tensor_map = TensorMap(
                keys=old_energy_tensor_map.keys,
                blocks=[new_block],
            )
            model_outputs[energy_target] = new_energy_tensor_map
        elif target_requires_strain_gradients:
            gradients = compute_gradient(
                model_outputs[energy_target].block().values,
                strains,
                is_training=is_training,
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

    return model_outputs


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
                        torch.tensor([i] * len(system))
                        for i, system in enumerate(gradients_list)
                    ]
                ),
                torch.concatenate(
                    [torch.arange(len(system)) for system in gradients_list]
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
        properties=Labels("energy", torch.tensor([[0]])).to(gradients.device),
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
        properties=Labels("energy", torch.tensor([[0]])).to(gradients.device),
    )


def _get_outputs(
    model: Union[torch.nn.Module, torch.jit._script.RecursiveScriptModule],
):
    if is_atomistic_model(model):
        return model.capabilities().outputs
    else:
        return model.outputs


def _get_model_outputs(
    model: Union[
        torch.nn.Module,
        MetatensorAtomisticModel,
        torch.jit._script.RecursiveScriptModule,
    ],
    systems: List[System],
    targets: Dict[str, TargetInfo],
    check_consistency: bool,
) -> Dict[str, TensorMap]:
    if is_atomistic_model(model):
        # put together an EvaluationOptions object
        options = ModelEvaluationOptions(
            length_unit="",  # this is only needed for unit conversions in MD engines
            outputs={
                key: ModelOutput(
                    quantity=value.quantity, unit=value.unit, per_atom=value.per_atom
                )
                for key, value in targets.items()
            },
        )
        return model(systems, options, check_consistency=check_consistency)
    else:
        return model(
            systems,
            {
                key: ModelOutput(
                    quantity=value.quantity, unit=value.unit, per_atom=value.per_atom
                )
                for key, value in targets.items()
            },
        )


@torch.jit.script
def _prepare_system(  # pragma: no cover
    system: System, positions_grad: bool, strain_grad: bool, check_consistency: bool
):
    """
    Prepares a system for gradient calculation.
    """
    if strain_grad:
        strain = torch.eye(
            3,
            dtype=system.cell.dtype,
            device=system.cell.device,
        ).requires_grad_(True)
        new_system = System(
            positions=system.positions @ strain,
            cell=system.cell @ strain,
            types=system.types,
            pbc=system.pbc,
        )
    else:
        if positions_grad:
            new_system = System(
                positions=system.positions.detach().clone().requires_grad_(True),
                cell=system.cell,
                types=system.types,
                pbc=system.pbc,
            )
            strain = None
        else:
            new_system = System(
                positions=system.positions,
                cell=system.cell,
                types=system.types,
                pbc=system.pbc,
            )
            strain = None

    for nl_options in system.known_neighbor_lists():
        nl = system.get_neighbor_list(nl_options)
        new_system.add_neighbor_list(nl_options, nl)

    return new_system, strain
