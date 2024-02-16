from typing import List, Union

import torch
from metatensor.learn.data.dataset import _BaseDataset
from metatensor.torch import TensorBlock, TensorMap


def get_average_number_of_atoms(
    datasets: List[Union[_BaseDataset, torch.utils.data.Subset]]
):
    """Calculate the average number of atoms in a dataset."""
    average_number_of_atoms = []
    for dataset in datasets:
        num_atoms = []
        for i in range(len(dataset)):
            structure = dataset[i].structure
            num_atoms.append(len(structure))
        average_number_of_atoms.append(
            torch.mean(torch.tensor(num_atoms).to(torch.get_default_dtype()))
        )
    return torch.tensor(average_number_of_atoms)


def get_average_number_of_neighbors(
    datasets: List[Union[_BaseDataset, torch.utils.data.Subset]]
):
    """Calculate the average number of neighbors in a dataset."""
    average_number_of_neighbors = []
    for dataset in datasets:
        num_neighbors = []
        for i in range(len(dataset)):
            structure = dataset[i].structure
            known_neighbors_lists = structure.known_neighbors_lists()
            if len(known_neighbors_lists) == 0:
                raise ValueError(
                    f"Structure {structure} does not have a neighbors list"
                )
            elif len(known_neighbors_lists) > 1:
                raise ValueError(
                    "More than one neighbors list per structure is not yet supported"
                )
            nl = structure.get_neighbors_list(known_neighbors_lists[0])
            num_neighbors.append(
                torch.mean(
                    torch.unique(nl.samples["first_atom"], return_counts=True)[1].to(
                        torch.get_default_dtype()
                    )
                )
            )
        average_number_of_neighbors.append(torch.mean(torch.tensor(num_neighbors)))
    return torch.tensor(average_number_of_neighbors)


def apply_normalization(
    atomic_property: TensorMap, normalization: torch.Tensor
) -> TensorMap:
    """Apply the normalization to an atomic property by dividing the
    atomic property by a normalization factor.

    Parameters
    ----------
    atomic_property: TensorMap
        Atomic property to apply the normalization to.
    normalization: torch.Tensor
        Normalization to apply.

    Returns
    -------
    TensorMap
        Atomic property with the normalization applied.
    """

    new_blocks: List[TensorBlock] = []
    for _, block in atomic_property.items():
        new_values = block.values / normalization
        new_blocks.append(
            TensorBlock(
                values=new_values,
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
        )

    return TensorMap(keys=atomic_property.keys, blocks=new_blocks)