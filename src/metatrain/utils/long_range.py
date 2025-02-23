from typing import List

import torch
from metatensor.torch.atomistic import System


class LongRangeFeaturizer(torch.nn.Module):
    def __init__(self, hypers, feature_dim, neighbor_list_options):
        super(LongRangeFeaturizer, self).__init__()

        try:
            from torchpme import InversePowerLawPotential
            from torchpme.calculators import (
                EwaldCalculator,
                P3MCalculator,
                PMECalculator,
            )
        except ImportError:
            raise ImportError(
                "`torch-pme` is required for long-range models. "
                "Please install it with `pip install torch-pme`."
            )

        if hypers["calculator"] == "ewald":
            self.calculator = EwaldCalculator(
                potential=InversePowerLawPotential(
                    exponent=hypers["exponent"],
                    smearing=hypers["atomic_smearing"],
                    exclusion_radius=neighbor_list_options.cutoff,
                ),
                full_neighbor_list=neighbor_list_options.full_list,
                lr_wavelength=hypers["lr_wavelength"],
                prefactor=hypers["prefactor"],
            )

        elif hypers["calculator"] == "p3m":
            self.calculator = P3MCalculator(
                potential=InversePowerLawPotential(
                    exponent=hypers["exponent"],
                    smearing=hypers["atomic_smearing"],
                    exclusion_radius=neighbor_list_options.cutoff,
                ),
                full_neighbor_list=neighbor_list_options.full_list,
                mesh_spacing=hypers["lr_wavelength"],
                prefactor=hypers["prefactor"],
            )
        elif hypers["calculator"] == "pme":
            self.calculator = PMECalculator(
                potential=InversePowerLawPotential(
                    exponent=hypers["exponent"],
                    smearing=hypers["atomic_smearing"],
                    exclusion_radius=neighbor_list_options.cutoff,
                ),
                full_neighbor_list=neighbor_list_options.full_list,
                mesh_spacing=hypers["lr_wavelength"],
                prefactor=hypers["prefactor"],
            )

        else:
            raise ValueError(
                f"Invalid torch-pme calculator: {hypers['calculator']}. "
                "Allowed options are ewald, pme, p3m."
            )

        self.neighbor_list_options = neighbor_list_options
        self.charges_map = torch.nn.Linear(feature_dim, feature_dim)

    def forward(
        self,
        systems: List[System],
        features: torch.Tensor,
        neighbor_distances: torch.Tensor,
    ) -> torch.Tensor:
        charges = self.charges_map(features)

        last_len_nodes = 0
        last_len_edges = 0
        potentials = []
        for system in systems:
            system_charges = charges[last_len_nodes : last_len_nodes + len(system)]
            last_len_nodes += len(system)

            neighbor_list = system.get_neighbor_list(self.neighbor_list_options)
            neighbor_indices_system = neighbor_list.samples.view(
                ["first_atom", "second_atom"]
            ).values

            neighbor_distances_system = neighbor_distances[
                last_len_edges : last_len_edges + len(neighbor_indices_system)
            ]
            last_len_edges += len(neighbor_indices_system)

            potential = self.calculator.forward(
                charges=system_charges,
                cell=system.cell,
                positions=system.positions,
                neighbor_indices=neighbor_indices_system,
                neighbor_distances=neighbor_distances_system,
            )
            potentials.append(potential * system_charges)
        return torch.cat(potentials)


class DummyLongRangeFeaturizer(torch.nn.Module):
    # a dummy class for torchscript
    def __init__(self):
        super().__init__()

    def forward(
        self,
        systems: List[System],
        features: torch.Tensor,
        neighbor_distances: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tensor(0)
