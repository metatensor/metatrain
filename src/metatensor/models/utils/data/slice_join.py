from typing import List

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


def slice(tensor_map: TensorMap, index: int):
    """
    Slices a TensorMap along a given dimension.

    :param tensor_map: The TensorMap to slice.
    :param index: The index to select.

    :return: The sliced TensorMap.
    """

    assert len(tensor_map) == 1
    assert len(tensor_map.block().samples.names) == 1
    assert tensor_map.block().samples.names == ["structure"]

    keys = tensor_map.keys
    samples = Labels(
        names=["structure"],
        values=torch.tensor([[index]]),
    )
    components = tensor_map.block().components
    properties = tensor_map.block().properties
    values = tensor_map.block().values[index].unsqueeze(0)

    block = TensorBlock(
        samples=samples,
        components=components,
        properties=properties,
        values=values,
    )

    for gradient_name, gradient_block in tensor_map.block().gradients():
        where_index = torch.where(gradient_block.samples.column("sample") == index)[0]

        gradient_sample_values = gradient_block.samples.values[where_index]
        gradient_sample_values[:, 0] = 0
        gradient_values = gradient_block.values[where_index]

        gradient_samples = Labels(
            names=gradient_block.samples.names,
            values=gradient_sample_values,
        )
        gradient_components = gradient_block.components
        gradient_properties = gradient_block.properties

        block.add_gradient(
            gradient_name,
            TensorBlock(
                values=gradient_values,
                samples=gradient_samples,
                components=gradient_components,
                properties=gradient_properties,
            ),
        )

    return TensorMap(
        keys=keys,
        blocks=[block],
    )


def join(tensor_map_list: List[TensorMap]):
    """
    Joins a list of TensorMaps into a single TensorMap.

    :param tensor_map_list: A list of TensorMaps.

    :return: A single TensorMap.
    """

    for tensor_map in tensor_map_list:
        assert len(tensor_map.blocks()) == 1
        assert len(tensor_map.block().samples) == 1

    keys = tensor_map_list[0].keys
    samples_names = tensor_map_list[0].block().samples.names
    samples_values = torch.cat(
        [tensor_map.block().samples.values for tensor_map in tensor_map_list]
    )
    components = tensor_map_list[0].block().components
    properties = tensor_map_list[0].block().properties
    values = torch.cat([tensor_map.block().values for tensor_map in tensor_map_list])

    block = TensorBlock(
        samples=Labels(names=samples_names, values=samples_values),
        components=components,
        properties=properties,
        values=values,
    )

    for gradient_name, gradient_block in tensor_map_list[0].block().gradients():
        gradient_values = torch.cat(
            [
                tensor_map.block().gradient(gradient_name).values
                for tensor_map in tensor_map_list
            ]
        )

        gradient_sample_values = []
        for index, tensor_map in enumerate(tensor_map_list):
            single_gradient_sample_values = (
                tensor_map.block().gradient(gradient_name).samples.values
            )
            single_gradient_sample_values[:, 0] = index  # update the "sample" value
            gradient_sample_values.append(single_gradient_sample_values)
        gradient_sample_values = torch.cat(gradient_sample_values)

        gradient_samples = Labels(
            names=gradient_block.samples.names,
            values=gradient_sample_values,
        )
        gradient_components = gradient_block.components
        gradient_properties = gradient_block.properties

        block.add_gradient(
            gradient_name,
            TensorBlock(
                values=gradient_values,
                samples=gradient_samples,
                components=gradient_components,
                properties=gradient_properties,
            ),
        )

    result = TensorMap(
        keys=keys,
        blocks=[block],
    )

    return result
