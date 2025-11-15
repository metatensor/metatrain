import ase.io
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


structures = ase.io.read("train_atoms_bmim_nod3.extxyz", ":")

all_forces = []
for atoms in structures:
    all_forces.append(torch.tensor(atoms.get_forces()))
all_forces = torch.cat(all_forces, dim=0)
all_forces = all_forces[:, [1, 2, 0]]

tensor_map = TensorMap(
    keys=Labels(
        names=["o3_lambda", "o3_sigma"],
        values=torch.tensor([[1, 1]]),
    ),
    blocks=[
        TensorBlock(
            values=all_forces.unsqueeze(-1),
            samples=Labels(
                names=["system", "atom"],
                values=torch.tensor(
                    [
                        [i, j]
                        for i, atoms in enumerate(structures)
                        for j in range(len(atoms))
                    ],
                    dtype=torch.long,
                ),
            ),
            components=[
                Labels(
                    names=["o3_mu"],
                    values=torch.tensor([[-1], [0], [1]], dtype=torch.long),
                )
            ],
            properties=Labels.single(),
        )
    ],
)
tensor_map.save("forces.mts")
