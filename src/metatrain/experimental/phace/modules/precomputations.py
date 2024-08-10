import sphericart.torch
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


class Precomputer(torch.nn.Module):
    def __init__(self, l_max):
        super().__init__()
        self.spherical_harmonics_split_list = [(2 * l + 1) for l in range(l_max + 1)]
        self.spherical_harmonics_calculator = sphericart.torch.SphericalHarmonics(
            l_max, normalized=True
        )

    def forward(
        self,
        positions,
        cells,
        species,
        cell_shifts,
        pairs,
        structure_pairs,
        structure_offsets,
    ):
        cartesian_vectors = get_cartesian_vectors(
            positions,
            cells,
            species,
            cell_shifts,
            pairs,
            structure_pairs,
            structure_offsets,
        )

        bare_cartesian_vectors = cartesian_vectors.values.squeeze(dim=-1)
        r = torch.sqrt((bare_cartesian_vectors**2).sum(dim=-1))

        spherical_harmonics = self.spherical_harmonics_calculator.compute(
            bare_cartesian_vectors
        )  # Get the spherical harmonics
        spherical_harmonics = spherical_harmonics * (4.0 * torch.pi) ** (
            0.5
        )  # normalize them
        spherical_harmonics = torch.split(
            spherical_harmonics, self.spherical_harmonics_split_list, dim=1
        )  # Split them into l chunks

        spherical_harmonics_blocks = [
            TensorBlock(
                values=spherical_harmonics_l.unsqueeze(-1),
                samples=cartesian_vectors.samples,
                components=[
                    Labels(
                        names=("o3_mu",),
                        values=torch.arange(
                            start=-l, end=l + 1, dtype=torch.int32
                        ).reshape(2 * l + 1, 1),
                    ).to(device=cartesian_vectors.values.device)
                ],
                properties=Labels(
                    names=["_"],
                    values=torch.zeros(
                        1, 1, dtype=torch.int32, device=cartesian_vectors.values.device
                    ),
                ),
            )
            for l, spherical_harmonics_l in enumerate(spherical_harmonics)
        ]
        spherical_harmonics_map = TensorMap(
            keys=Labels(
                names=["o3_lambda"],
                values=torch.arange(
                    len(spherical_harmonics_blocks), device=r.device
                ).reshape(len(spherical_harmonics_blocks), 1),
            ),
            blocks=spherical_harmonics_blocks,
        )

        r_block = TensorBlock(
            values=r.unsqueeze(-1),
            samples=cartesian_vectors.samples,
            components=[],
            properties=Labels(
                names=["_"],
                values=torch.zeros(1, 1, dtype=torch.int32, device=r.device),
            ),
        )

        return r_block, spherical_harmonics_map


def get_cartesian_vectors(
    positions, cells, species, cell_shifts, pairs, structure_pairs, structure_offsets
):
    """
    Wraps direction vectors into TensorBlock object with metadata information
    """

    # calculate interatomic vectors
    pairs_offsets = structure_offsets[structure_pairs]
    shifted_pairs = pairs_offsets[:, None] + pairs
    shifted_pairs_i = shifted_pairs[:, 0]
    shifted_pairs_j = shifted_pairs[:, 1]
    direction_vectors = - (
        positions[shifted_pairs_j]
        - positions[shifted_pairs_i] 
        + torch.einsum(
            "ab, abc -> ac", cell_shifts.to(cells.dtype), cells[structure_pairs]
        )
    )

    # find associated metadata
    pairs_i = pairs[:, 0]
    pairs_j = pairs[:, 1]
    labels = torch.stack(
        [
            structure_pairs,
            pairs_i,
            pairs_j,
            species[shifted_pairs_i],
            species[shifted_pairs_j],
            cell_shifts[:, 0],
            cell_shifts[:, 1],
            cell_shifts[:, 2],
        ],
        dim=-1,
    )

    # build TensorBlock
    block = TensorBlock(
        values=direction_vectors.unsqueeze(dim=-1),
        samples=Labels(
            names=[
                "structure",
                "center",
                "neighbor",
                "species_center",
                "species_neighbor",
                "cell_x",
                "cell_y",
                "cell_z",
            ],
            values=labels,
        ),
        components=[
            Labels(
                names=["cartesian_dimension"],
                values=torch.tensor([-1, 0, 1], dtype=torch.int32).reshape((-1, 1)),
            ).to(device=direction_vectors.device)
        ],
        properties=Labels(
            names=["_"],
            values=torch.zeros(
                1, 1, dtype=torch.int32, device=direction_vectors.device
            ),
        ),
    )

    return block
