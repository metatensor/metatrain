import ase.io
import torch
import tqdm
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import NeighborListOptions, systems_to_torch

from metatrain.utils.data import DiskDatasetWriter
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


disk_dataset_writer = DiskDatasetWriter("qm9_reduced_100.zip")
for i in tqdm.tqdm(range(100)):
    frame = ase.io.read("qm9_reduced_100.xyz", index=i)
    system = systems_to_torch(frame, dtype=torch.float64)
    system = get_system_with_neighbor_lists(
        system,
        [NeighborListOptions(cutoff=5.0, full_list=True, strict=True)],
    )
    energy = TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.tensor([[frame.info["U0"]]], dtype=torch.float64),
                samples=Labels(
                    names=["system"],
                    values=torch.tensor([[i]]),
                ),
                components=[],
                properties=Labels("energy", torch.tensor([[0]])),
            )
        ],
    )
    disk_dataset_writer.write_sample(system, {"energy": energy})
del disk_dataset_writer
