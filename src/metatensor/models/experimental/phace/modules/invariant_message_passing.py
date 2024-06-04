from typing import Dict, List

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from .normalize import Normalizer
from .radial_basis import RadialBasis


class InvariantMessagePasser(torch.nn.Module):

    def __init__(self, hypers: Dict, all_species: List[int]) -> None:
        super().__init__()

        self.all_species = all_species
        hypers["radial_basis"]["r_cut"] = hypers["cutoff"]
        hypers["radial_basis"]["normalize"] = hypers["normalize"]
        hypers["radial_basis"]["n_element_channels"] = hypers["n_element_channels"]
        self.radial_basis_calculator = RadialBasis(hypers["radial_basis"], all_species)
        self.n_max_l = self.radial_basis_calculator.n_max_l
        self.k_max_l = [hypers["n_element_channels"] * n_max for n_max in self.n_max_l]
        self.l_max = len(self.n_max_l) - 1
        self.pooling_normalization = torch.nn.ModuleList(
            [Normalizer([0, 1, 2]) for l in range(self.l_max + 1)]
        )
        self.irreps_out = [(l, 1) for l in range(self.l_max + 1)]

    def forward(
        self,
        r: TensorBlock,
        sh: TensorMap,
        centers,
        neighbors,
        n_atoms: int,
        initial_center_embedding: TensorMap,
        samples: Labels,  # TODO: can this go?
    ) -> TensorMap:

        # TODO: extract radial basis calculation to a separate module
        # (e.g. vector expansion) and use the splines once
        radial_basis = self.radial_basis_calculator(r.values.squeeze(-1), r.samples)

        labels: List[List[int]] = []
        blocks: List[TensorBlock] = []
        for l, normalizer_l in enumerate(self.pooling_normalization):
            spherical_harmonics_l = sh.block({"o3_lambda": l}).values
            radial_basis_l = radial_basis[l]
            densities_l = torch.zeros(
                (n_atoms, spherical_harmonics_l.shape[1], radial_basis_l.shape[1]),
                device=radial_basis_l.device,
                dtype=radial_basis_l.dtype,
            )
            densities_l.index_add_(
                dim=0,
                index=centers,
                source=spherical_harmonics_l
                * radial_basis_l.unsqueeze(1)
                * initial_center_embedding.block().values[neighbors][
                    :, :, : radial_basis_l.shape[1]
                ],
            )
            densities_l = normalizer_l(densities_l)
            labels.append([1, l, 1])
            blocks.append(
                TensorBlock(
                    values=densities_l,
                    samples=samples,
                    components=sh.block({"o3_lambda": l}).components,
                    properties=Labels(
                        "properties",
                        torch.arange(
                            densities_l.shape[2],
                            dtype=torch.int,
                            device=densities_l.device,
                        ).unsqueeze(-1),
                    ),
                )
            )

        return TensorMap(
            keys=Labels(
                names=["nu", "o3_lambda", "o3_sigma"],
                values=torch.tensor(labels, dtype=torch.int32),
            ).to(device=densities_l.device),
            blocks=blocks,
        )
