"""Utilities related to the interaction with metatensor"""

from typing import Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.operations._add import _add_block_block
from metatomic.torch import ModelOutput, System

from metatrain.utils.additive import CompositionModel
from metatrain.utils.data import TargetInfo
from metatrain.utils.sum_over_atoms import sum_over_atoms


def add_contribution(
    values: Dict[str, TensorMap],
    systems: List[System],
    outputs: Dict[str, ModelOutput],
    additive_model: CompositionModel,
    selected_atoms: Optional[Labels] = None,
) -> None:
    outputs_for_additive_model: Dict[str, ModelOutput] = {}
    for name, output in outputs.items():
        if name in additive_model.outputs:
            outputs_for_additive_model[name] = output
    additive_contributions = additive_model.forward(
        systems,
        outputs_for_additive_model,
        selected_atoms,
    )
    for name in additive_contributions:
        # # TODO: uncomment this after metatensor.torch.add is updated to
        # # handle sparse sums
        # return_dict[name] = metatensor.torch.add(
        #     return_dict[name],
        #     additive_contributions[name].to(
        #         device=return_dict[name].device,
        #         dtype=return_dict[name].dtype
        #         ),
        # )

        # TODO: "manual" sparse sum: update to metatensor.torch.add after
        # sparse sum is implemented in metatensor.operations
        output_blocks: List[TensorBlock] = []
        for k, b in values[name].items():
            if k in additive_contributions[name].keys:
                output_blocks.append(
                    _add_block_block(
                        b,
                        additive_contributions[name]
                        .block(k)
                        .to(device=b.device, dtype=b.dtype),
                    )
                )
            else:
                output_blocks.append(b)
        values[name] = TensorMap(values[name].keys, output_blocks)


def e3nn_to_tensormap(
    target_values: torch.Tensor,
    sample_labels: Labels,
    target_info: TargetInfo,
    output_name: str,
    outputs: Dict[str, ModelOutput],
) -> TensorMap:
    blocks: list[TensorBlock] = []
    pointer = 0
    for i in range(len(target_info.component_labels)):
        components = target_info.component_labels[i]
        properties = target_info.property_labels[i]

        has_components = len(components) > 0
        n_components = len(components[0]) if has_components else 1
        n_properties = len(properties)

        end = pointer + n_components * n_properties

        values = (
            target_values[:, pointer:end]
            .reshape(
                -1,
                n_properties,
                n_components,
            )
            .transpose(1, 2)
        )

        if target_info.is_cartesian and n_components == 3:
            # Go back from YZX to XYZ
            values = values[:, [2, 0, 1], :]

        if not has_components:
            # Remove the components dimension if there are no components
            values = values.squeeze(1)

        blocks.append(
            TensorBlock(
                values=values,
                samples=sample_labels,
                components=components,
                properties=properties,
            )
        )
        pointer = end

    atom_target = TensorMap(keys=target_info.layout.keys, blocks=blocks)

    return (
        sum_over_atoms(atom_target)
        if not outputs[output_name].per_atom
        else atom_target
    )


def get_system_indices_and_labels(systems: List[System]) -> tuple[torch.Tensor, Labels]:
    device = systems[0].device

    system_indices = torch.concatenate(
        [
            torch.full(
                (len(system),),
                i_system,
                device=device,
            )
            for i_system, system in enumerate(systems)
        ],
    )

    sample_values = torch.stack(
        [
            system_indices,
            torch.concatenate(
                [
                    torch.arange(
                        len(system),
                        device=device,
                    )
                    for system in systems
                ],
            ),
        ],
        dim=1,
    )
    sample_labels = Labels(
        names=["system", "atom"],
        values=sample_values,
    )
    return system_indices, sample_labels
