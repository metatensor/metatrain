from typing import List, Union

import torch
from metatensor.learn.data.dataset import _BaseDataset
from metatensor.torch import Labels, TensorBlock
from metatensor.torch.atomistic import (
    NeighborsListOptions,
    System,
    register_autograd_neighbors,
)
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
    """
    Calculates the neighborlist for a given system using
    rascaline.NeghborList calculator. Additionally, it registers
    the neighborlist for torch autograd.

    :param systems: A single systems or a list systems.
    :param options: A NeighborsListOptions object.

    :return: A TensorBlock or a list of TensorBlocks containing neigborlists
    information.
    """

    if not isinstance(systems, list):
        systems = [systems]
    nl_calculator = NeighborList(
        cutoff=options.model_cutoff, full_neighbor_list=options.full_list
    )
    nl_list = []
    for system in systems:
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


def check_and_update_neighbors_lists(
    datasets: Union[_BaseDataset, List[_BaseDataset]],
    requested_neighbors_lists: List[NeighborsListOptions],
):
    """
    This is a helper function that checks that the neighborlists in the datasets
    are compatible with the model's requested neighborlists and adds the
    missing neighborlists.

    :param datasets: A single dataset or a list of datasets.
    :param requested_neighbors_lists: A list of NeighborsListOptions requested
    by the model.
    """
    if not isinstance(datasets, list):
        datasets = [datasets]
    for dataset in datasets:
        for i in range(len(dataset)):
            system = dataset[i].structure
            known_neighbors_lists = system.known_neighbors_lists()
            for nl_options in requested_neighbors_lists:
                if nl_options not in known_neighbors_lists:
                    nl = get_rascaline_neighbors_list(
                        system,
                        nl_options,
                    )
                    register_autograd_neighbors(system, nl)
                    system.add_neighbors_list(nl_options, nl)
