"""Utilities related to the interaction with metatensor"""

from typing import Dict, List, Optional

import torch
from e3nn import o3
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.operations._add import _add_block_block
from metatomic.torch import ModelOutput, System

from metatrain.utils.additive import CompositionModel
from metatrain.utils.data import TargetInfo
from metatrain.utils.data.target_info import get_generic_target_info


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
    samples: Labels,
    target_info: TargetInfo,
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
                samples=samples,
                components=components,
                properties=properties,
            )
        )
        pointer = end

    return TensorMap(keys=target_info.layout.keys, blocks=blocks)


def get_e3nn_target_info(target_name: str, target: dict) -> TargetInfo:
    """Get the target info corresponding to some e3nn irreps.

    :param target_name: Name of the target.
    :param target: Target dictionary containing the irreps and other info.
    :return: The corresponding TargetInfo object.
    """
    irreps = o3.Irreps(target["irreps"])
    return get_generic_target_info(
        target_name,
        {
            "quantity": target.get("quantity", ""),
            "unit": target.get("unit", ""),
            "type": {
                "spherical": {
                    "irreps": [
                        {"o3_lambda": ir.ir.l, "o3_sigma": ir.ir.p * ((-1) ** ir.ir.l)}
                        for ir in irreps
                    ]
                }
            },
            "num_subtargets": [ir.mul for ir in irreps],
            "per_atom": target["per_atom"],
        },
    )


def target_info_to_e3nn_irreps(target_info: TargetInfo) -> o3.Irreps:
    """Convert a TargetInfo to e3nn Irreps.

    :param target_info: TargetInfo object.
    :return: e3nn Irreps corresponding to the TargetInfo.
    """
    irreps = []
    for key, block in target_info.layout.items():
        multiplicity = len(block.properties.values)

        if target_info.is_scalar:
            irreps.append((multiplicity, (0, 1)))
        elif target_info.is_spherical:
            ell = int(key["o3_lambda"])
            irreps.append((multiplicity, (ell, (-1) ** ell)))
        elif target_info.is_cartesian:
            ell = 1
            irreps.append((multiplicity, (ell, (-1) ** ell)))
    return o3.Irreps(irreps)


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
