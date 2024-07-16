import os

import torch
from metatensor.torch.atomistic import NeighborListOptions

from metatrain.utils.data.get_dataset import get_dataset
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


HERE = os.path.dirname(os.path.abspath(__file__))
dataset_folder = os.path.join(HERE, "ethanol_disk/")

config = {
    "systems": {"read_from": "ethanol_reduced_100.xyz", "reader": "ase"},
    "targets": {
        "energy": {
            "quantity": "energy",
            "read_from": "ethanol_reduced_100.xyz",
            "reader": "ase",
            "key": "energy",
            "unit": "kcal/mol",
            "forces": {
                "read_from": "ethanol_reduced_100.xyz",
                "reader": "ase",
                "key": "forces",
            },
            "stress": False,
            "virial": False,
        },
    }
}
dataset, _ = get_dataset(config)
requested_neighbor_list = NeighborListOptions(cutoff=5.0, full_list=False)

if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

for index, sample in enumerate(dataset):
    get_system_with_neighbor_lists(sample["system"], [requested_neighbor_list])
    torch.save(sample, os.path.join(dataset_folder, f"sample_{index}.mts"))
    # free some memory
    del sample["system"]
    del sample["energy"]
    del dataset[index]["system"]
    del dataset[index]["energy"]
