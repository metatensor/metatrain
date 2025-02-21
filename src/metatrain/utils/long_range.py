from typing import List

import torch
from metatensor.torch.atomistic import NeighborListOptions, System

from torchpme.calculators import EwaldCalculator, P3MCalculator
from torchpme import InversePowerLawPotential

class LongRangeFeaturizer(torch.nn.Module):
    def __init__(self, hypers):
        super(LongRangeFeaturizer, self).__init__()

        if hypers["long_range"]["calculator"] == "ewald":
            self.calculator = EwaldCalculator(
                potential=InversePowerLawPotential(
                    exponent=hypers["long_range"]["exponent"],
                    smearing=hypers["long_range"]["atomic_smearing"], 
                    exclusion_radius=hypers["cutoff"]),
                full_neighbor_list=True,
                lr_wavelength=hypers["long_range"]["lr_wavelength"],
                prefactor=hypers["long_range"]["prefactor"],    
            )

        elif hypers["long_range"]["calculator"] == "p3m":
            self.calculator = P3MCalculator(
                potential=InversePowerLawPotential(
                    exponent=hypers["long_range"]["exponent"],
                    smearing=hypers["long_range"]["atomic_smearing"], 
                    exclusion_radius=hypers["cutoff"]),
                full_neighbor_list=True,
                mesh_spacing=hypers["long_range"]["lr_wavelength"],
                prefactor=hypers["long_range"]["prefactor"],    
            )

        self.charges_map = torch.nn.Linear(hypers["d_pet"], hypers["d_pet"])

    def forward(
        self,
        systems: List[System],
        features: torch.Tensor,
        neighbor_list_options: NeighborListOptions) -> torch.Tensor:

        charges = self.charges_map(features)

        last_len = 0
        potentials = []
        for i, system in enumerate(systems):
            system_charges = charges[last_len:last_len+len(system)]
            last_len += len(system)

            neighbor_list = system.get_neighbor_list(neighbor_list_options)
            neighbor_indices = neighbor_list.samples.view(["first_atom", "second_atom"]).values
            neighbor_distances = torch.linalg.norm(neighbor_list.values, dim=1).squeeze(1)

            potential = self.calculator.forward(
                charges=system_charges,
                cell=system.cell,
                positions=system.positions,
                neighbor_indices=neighbor_indices,
                neighbor_distances=neighbor_distances,
            )
            potentials.append(potential * system_charges)
        return torch.cat(potentials)
