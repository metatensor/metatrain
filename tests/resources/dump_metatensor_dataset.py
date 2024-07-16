import os

import torch
from metatensor.torch.atomistic import NeighborListOptions

from metatrain.utils.data import Dataset, read_systems, read_targets
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


HERE = os.path.dirname(os.path.abspath(__file__))
dataset_folder = os.path.join(HERE, "qm9_disk/")

qm9_systems = read_systems("qm9_reduced_100.xyz")
target_config = {
    "energy": {
        "quantity": "energy",
        "read_from": "qm9_reduced_100.xyz",
        "file_format": ".xyz",
        "key": "U0",
        "unit": "hartree",
        "forces": False,
        "stress": False,
        "virial": False,
    },
}
targets, _ = read_targets(target_config)
requested_neighbor_list = NeighborListOptions(cutoff=5.0, full_list=False)
systems = [
    get_system_with_neighbor_lists(system, [requested_neighbor_list])
    for system in qm9_systems
]
dataset = Dataset({"system": qm9_systems, **targets})

if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)
for index, sample in enumerate(dataset):
    torch.save(sample, os.path.join(dataset_folder, f"sample_{index}.mts"))
