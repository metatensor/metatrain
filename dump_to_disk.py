import os
import zipfile
import io

import torch

from metatensor.torch import Labels, TensorBlock
from metatensor.torch.atomistic import NeighborListOptions, System, save, load_system
import tqdm


system = System(
    types=torch.tensor([1, 2, 3, 4]),
    positions=torch.rand((4, 3), dtype=torch.float64),
    cell=torch.rand((3, 3), dtype=torch.float64),
    pbc=torch.tensor([True, True, True], dtype=torch.bool),
)
nl_block = TensorBlock(
    values=torch.rand(2000, 3, 1, dtype=torch.float64),
    samples=Labels(
        [
            "first_atom",
            "second_atom",
            "cell_shift_a",
            "cell_shift_b",
            "cell_shift_c",
        ],
        torch.arange(2000 * 5, dtype=torch.int64).reshape(2000, 5),
    ),
    components=[Labels.range("xyz", 3)],
    properties=Labels.range("distance", 1),
)
system.add_neighbor_list(
    NeighborListOptions(cutoff=3.5, full_list=True, strict=True),
    nl_block,
)


with zipfile.ZipFile(os.path.join("dump", "systems.zip"), "w") as zipf:
    for i in tqdm.tqdm(range(10000)):
        with zipf.open(f"system_{i}.mta", "w") as file:
            save(file, system)

with zipfile.ZipFile(os.path.join("dump", "systems.zip"), "r") as zipf:
    for i in tqdm.tqdm(range(10000)):
        with zipf.open(f"system_{i}.mta", "r") as file:
            system = load_system(file)
