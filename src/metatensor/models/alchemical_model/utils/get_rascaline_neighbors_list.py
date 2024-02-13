from typing import List, Union

import torch
from metatensor.torch import Labels, TensorBlock
from metatensor.torch.atomistic import NeighborsListOptions, System
from rascaline.torch import NeighborList


REQUIRED_NL_SAMPLES = [
    "first_atom",
    "second_atom",
    "cell_shift_a",
    "cell_shift_b",
    "cell_shift_c",
]


def get_rascaline_neighbors_list(
    systems: Union[System, List[System]], options: NeighborsListOptions
) -> Union[TensorBlock, List[TensorBlock]]:
    if not isinstance(systems, list):
        systems = [systems]
    nl_list = []
    for system in systems:
        nl_calculator = NeighborList(
            cutoff=options.model_cutoff, full_neighbor_list=options.full_list
        )
        nl_tmap = nl_calculator.compute(system)
        tmp_nl = nl_tmap.keys_to_samples(nl_tmap.keys.names).block()
        required_indices = [
            i
            for i in range(len(tmp_nl.samples.names))
            if tmp_nl.samples.names[i] in REQUIRED_NL_SAMPLES
        ]
        samples = Labels(
            names=REQUIRED_NL_SAMPLES, values=tmp_nl.samples.values[:, required_indices]
        )
        components = Labels(names=["xyz"], values=tmp_nl.components[0].values)
        properties = Labels(
            names=tmp_nl.properties.names,
            values=torch.zeros_like(tmp_nl.properties.values),
        )
        nl = TensorBlock(
            samples=samples,
            components=[components],
            properties=properties,
            values=tmp_nl.values,
        )
        nl_list.append(nl)
    if len(nl_list) == 1:
        return nl_list[0]
    else:
        return nl_list
