from typing import List

import torch
from metatensor.torch.atomistic import System


class LongRangeFeaturizer(torch.nn.Module):
    """A class to compute long-range features starting from short-range features.

    :param hypers: Dictionary containing the hyperparameters for the long-range
        featurizer.
    :param feature_dim: The dimension of the short-range features (which also
        corresponds to the number of long-range features that will be returned).
    :param neighbor_list_options: A :py:class:`NeighborListOptions` object containing
        the neighbor list information for the short-range model.
    """

    def __init__(self, hypers, feature_dim, neighbor_list_options):
        super(LongRangeFeaturizer, self).__init__()

        try:
            from torchpme import CoulombPotential, InversePowerLawPotential
            from torchpme.calculators import Calculator, P3MCalculator
        except ImportError:
            raise ImportError(
                "`torch-pme` is required for long-range models. "
                "Please install it with `pip install torch-pme`."
            )

        if hypers["exponent"] == 1:
            self.calculator = P3MCalculator(
                potential=CoulombPotential(
                    smearing=1.4,
                    exclusion_radius=neighbor_list_options.cutoff,
                ),
                interpolation_nodes=5,
                full_neighbor_list=neighbor_list_options.full_list,
                mesh_spacing=1.33,
            )
            self.direct_calculator = Calculator(
                potential=CoulombPotential(
                    smearing=None,
                    exclusion_radius=neighbor_list_options.cutoff,
                ),
                full_neighbor_list=False,
            )
        else:
            self.calculator = P3MCalculator(
                potential=InversePowerLawPotential(
                    exponent=hypers["exponent"],
                    smearing=1.4,
                    exclusion_radius=neighbor_list_options.cutoff,
                ),
                interpolation_nodes=5,
                full_neighbor_list=neighbor_list_options.full_list,
                mesh_spacing=1.33,
            )
            self.direct_calculator = Calculator(
                potential=InversePowerLawPotential(
                    exponent=hypers["exponent"],
                    smearing=None,
                    exclusion_radius=neighbor_list_options.cutoff,
                ),
                full_neighbor_list=False,
            )

        self.neighbor_list_options = neighbor_list_options
        self.charges_map = torch.nn.Linear(feature_dim, feature_dim)

    def forward(
        self,
        systems: List[System],
        features: torch.Tensor,
        neighbor_distances: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the long-range features for a list of systems.

        :param systems: A list of :py:class:`System` objects for which to compute the
            long-range features. Each system must contain a neighbor list consistent
            with the neighbor list options used to create the class.
        :param features: A tensor of short-range features for the systems.
        :param neighbor_distances: A tensor of neighbor distances for the systems,
            which must be consistent with the neighbor list options used to create the
            class.
        """
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

            if not system.pbc.all():
                neighbor_indices_system = torch.combinations(
                    torch.arange(len(system)), 2
                ).to(system.positions.device)
                neighbor_distances_system = torch.norm(
                    system.positions[neighbor_indices_system[:, 0]]
                    - system.positions[neighbor_indices_system[:, 1]],
                    dim=1,
                ).to(system.positions.device)
                potential = self.direct_calculator.forward(
                    charges=system_charges,
                    cell=system.cell,
                    positions=system.positions,
                    neighbor_indices=neighbor_indices_system,
                    neighbor_distances=neighbor_distances_system,
                )
            else:
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
