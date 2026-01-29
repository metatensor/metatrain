"""Utilities related to the interaction with metatensor.

These utilities are mainly focused on helping wrapping the outputs
of MACE (raw torch tensors that correspond to some e3nn irreps) to
metatensor TensorMaps.
"""

from typing import List

import torch
from e3nn import o3
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from metatrain.utils.data import TargetInfo


def e3nn_to_tensormap(
    target_values: torch.Tensor,
    samples: Labels,
    layout: TensorMap,
) -> TensorMap:
    """Convert the e3nn torch tensor outputs to a metatensor TensorMap.

    :param target_values: Tensor of shape ``(n_samples, dim)``
        containing the target values, with ``dim`` being the flattened
        dimension containing all irreps.
    :param samples: ``Labels`` for the samples argument of ``TensorBlock``.
    :param layout: ``TensorMap`` object containing the layout of the
        e3nn irreps. This has probably been obtained with
        ``get_e3nn_target_info``.

    :return: A ``TensorMap`` containing the target values.
    """
    # Check if this is a cartesian target
    is_cartesian = (
        len(layout) > 0
        and len(layout.block(0).components) > 0
        and layout.block(0).components[0].names[0].startswith("xyz")
    )

    blocks: list[TensorBlock] = []
    pointer = 0
    for block in layout.blocks():
        components = block.components
        properties = block.properties

        has_components = len(components) > 0
        n_components = len(components[0]) if has_components else 1
        n_properties = len(properties)

        end = pointer + n_components * n_properties

        values = (
            target_values[:, pointer:end]
            .reshape(
                target_values.shape[0],
                n_properties,
                n_components,
            )
            .transpose(1, 2)
        )

        if is_cartesian and n_components == 3:
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

    return TensorMap(keys=layout.keys, blocks=blocks)


def get_e3nn_mts_layout(target_name: str, target: dict) -> TensorMap:
    """Get the tensormap layout corresponding to some e3nn irreps.

    This function follows the API of the ``get_generic_target_info``
    function of ``metatrain.utils.data.target_info``, because at some
    point it can be converted to return a ``TargetInfo``.

    :param target_name: Name of the target.
    :param target: Target dictionary containing the irreps and other info.
    :return: The corresponding ``TensorMap`` object.
    """
    sample_names = ["system"]
    if target["per_atom"]:
        sample_names.append("atom")

    properties_name = target.get("properties_name", target_name.replace("mtt::", ""))

    irreps = o3.Irreps(target["type"]["spherical"]["irreps"])
    keys = []
    blocks = []
    for i, irrep in enumerate(irreps):
        o3_lambda = irrep.ir.l
        o3_sigma = irrep.ir.p * ((-1) ** o3_lambda)
        num_properties = irrep.mul

        components = [
            Labels(
                names=["o3_mu"],
                values=torch.arange(
                    -o3_lambda, o3_lambda + 1, dtype=torch.int32
                ).reshape(-1, 1),
            )
        ]
        block = TensorBlock(
            # float64: otherwise metatensor can't serialize
            values=torch.empty(
                0,
                2 * o3_lambda + 1,
                num_properties,
                dtype=torch.float64,
            ),
            samples=Labels(
                names=sample_names,
                values=torch.empty((0, len(sample_names)), dtype=torch.int32),
            ),
            components=components,
            properties=Labels.range(properties_name, num_properties),
        )
        keys.append([o3_lambda, o3_sigma, i])
        blocks.append(block)

    layout = TensorMap(
        keys=Labels(
            ["o3_lambda", "o3_sigma", "i_irrep"], torch.tensor(keys, dtype=torch.int32)
        ),
        blocks=blocks,
    )

    return layout


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
            p = int(key["o3_sigma"] * ((-1) ** ell))
            irreps.append((multiplicity, (ell, p)))
        elif target_info.is_cartesian:
            ell = 1
            irreps.append((multiplicity, (ell, (-1) ** ell)))
    return o3.Irreps(irreps)


def get_samples_labels(systems: List[System]) -> Labels:
    """For a list of systems, get the Labels object for the samples
    of a ``TensorBlock`` containing per-atom data.

    :param systems: List of systems.
    :return: ``Labels`` object for the samples of a ``TensorBlock``.
    """
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
    return sample_labels
