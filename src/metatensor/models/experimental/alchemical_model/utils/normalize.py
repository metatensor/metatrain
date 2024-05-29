from typing import List, Union

import metatensor.torch
import torch
import torch.bin

from metatensor.models.utils.data import Dataset


def get_average_number_of_atoms(
    datasets: List[Union[Dataset, torch.utils.data.Subset]]
):
    """Calculates the average number of atoms in a dataset.

    :param datasets: A list of datasets.

    :return: A `torch.Tensor` object with the average number of atoms.
    """
    average_number_of_atoms = []
    for dataset in datasets:
        dtype = dataset[0]["system"].positions.dtype
        num_atoms = []
        for i in range(len(dataset)):
            system = dataset[i]["system"]
            num_atoms.append(len(system))
        average_number_of_atoms.append(torch.mean(torch.tensor(num_atoms, dtype=dtype)))
    return torch.tensor(average_number_of_atoms)


def get_average_number_of_neighbors(
    datasets: List[Union[Dataset, torch.utils.data.Subset]]
) -> torch.Tensor:
    """Calculate the average number of neighbor in a dataset.

    :param datasets: A list of datasets.

    :return: A `torch.Tensor` object with the average number of neighbor.
    """
    average_number_of_neighbors = []
    for dataset in datasets:
        num_neighbor = []
        dtype = dataset[0]["system"].positions.dtype
        for i in range(len(dataset)):
            system = dataset[i]["system"]
            known_neighbor_lists = system.known_neighbor_lists()
            if len(known_neighbor_lists) == 0:
                raise ValueError(f"system {system} does not have a neighbor list")
            elif len(known_neighbor_lists) > 1:
                raise ValueError(
                    "More than one neighbor list per system is not yet supported"
                )
            nl = system.get_neighbor_list(known_neighbor_lists[0])
            num_neighbor.append(
                torch.mean(
                    torch.unique(nl.samples["first_atom"], return_counts=True)[1].to(
                        dtype
                    )
                )
            )
        average_number_of_neighbors.append(torch.mean(torch.tensor(num_neighbor)))
    return torch.tensor(average_number_of_neighbors)


def remove_composition_from_dataset(
    dataset: Union[Dataset, torch.utils.data.Subset],
    all_species: List[int],
    composition_weights: torch.Tensor,
) -> List[Union[Dataset, torch.utils.data.Subset]]:
    """Remove the composition from the dataset.

    :param datasets: A list of datasets.

    :return: A list of datasets with the composition contribution removed.
    """
    # assert one property
    first_sample = next(iter(dataset))
    assert len(first_sample) == 2  # system and property
    property_name = list(first_sample.keys())[1]

    new_systems = []
    new_properties = []
    # remove composition from dataset
    for i in range(len(dataset)):
        system = dataset[i]["system"]
        property = dataset[i][property_name]
        numbers = system.types
        composition = torch.bincount(numbers, minlength=max(all_species) + 1)
        composition = composition[all_species].to(
            device=composition_weights.device, dtype=composition_weights.dtype
        )
        property = metatensor.torch.subtract(
            property, torch.dot(composition, composition_weights).item()
        )
        new_systems.append(system)
        new_properties.append(property)

    new_dataset = Dataset({"system": new_systems, property_name: new_properties})
    return new_dataset
