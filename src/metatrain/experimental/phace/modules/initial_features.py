import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


def get_initial_features(structures, centers, species, dtype: torch.dtype, n_features: int):

    n_atoms = structures.shape[0]
    block = TensorBlock(
        values=torch.ones(
            (n_atoms, 1, n_features), dtype=dtype, device=structures.device
        ),
        samples=Labels(
            names=["system", "atom", "center_type"],
            values=torch.stack([structures, centers, species], dim=1),
        ).to(device=structures.device),
        components=[
            Labels(
                names=["o3_mu"],
                values=torch.tensor([[0]], device=structures.device),
            ).to(device=structures.device)
        ],
        properties=Labels(
            "properties",
            torch.arange(
                n_features, dtype=torch.int, device=structures.device
            ).unsqueeze(-1),
        ),
    )
    return TensorMap(
        keys=Labels(
            names=["nu", "o3_lambda", "o3_sigma"],
            values=torch.tensor([[0, 0, 1]], device=structures.device),
        ),
        blocks=[block],
    )
