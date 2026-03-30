# mypy: disable-error-code=misc
# We ignore misc errors in this file because TypedDict
# with default values is not allowed by mypy.
from typing import Optional

import torch
from metatomic.torch import System
from torch.nn.utils.rnn import pad_sequence
from typing_extensions import TypedDict

from metatrain.utils.neighbor_lists import NeighborListOptions


class LongRangeHypers(TypedDict):
    """In some systems and datasets, enabling long-range Coulomb interactions
    might be beneficial for the accuracy of the model and/or
    its physical correctness."""

    enable: bool = False
    """Toggle for enabling long-range interactions"""
    use_ewald: bool = True
    """Use Ewald summation. If False, P3M is used"""
    smearing: float = 1.4
    """Smearing width in Fourier space"""
    kspace_resolution: float = 1.33
    """Resolution of the reciprocal space grid"""
    interpolation_nodes: int = 5
    """Number of grid points for interpolation (for PME only)"""
    num_charges: Optional[int] = None
    """Number of latent charge channels used by the long-range module. If set to
    ``None``, the short-range feature dimension is used."""
    prefactor: float = 1.5
    """Prefactor for the long-range potential features."""
    every_layer: bool = False
    """Whether to compute long-range features at every layer of the model, or only at
    the end."""


class LongRangeFeaturizer(torch.nn.Module):
    """A class to compute long-range features starting from short-range features.

    :param hypers: Dictionary containing the hyperparameters for the long-range
        featurizer.
    :param feature_dim: The dimension of the short-range features (which also
        corresponds to the number of long-range features that will be returned).
    :param neighbor_list_options: A :py:class:`NeighborListOptions` object containing
        the neighbor list information for the short-range model.
    """

    def __init__(
        self,
        hypers: LongRangeHypers,
        feature_dim: int,
        neighbor_list_options: NeighborListOptions,
    ) -> None:
        super(LongRangeFeaturizer, self).__init__()

        try:
            from torchpme import (
                Calculator,
                CoulombPotential,
                EwaldCalculator,
                P3MCalculator,
            )
        except ImportError:
            raise ImportError(
                "`torch-pme` is required for long-range models. "
                "Please install it with `pip install 'torch-pme>=0.3.2'`."
            )

        self.prefactor = float(hypers["prefactor"])
        num_charges = hypers["num_charges"]
        self.num_charges = feature_dim if num_charges is None else int(num_charges)

        self.ewald_calculator = EwaldCalculator(
            potential=CoulombPotential(
                smearing=float(hypers["smearing"]),
                exclusion_radius=neighbor_list_options.cutoff,
            ),
            full_neighbor_list=neighbor_list_options.full_list,
            lr_wavelength=float(hypers["kspace_resolution"]),
        )
        """Calculator to compute the long-range electrostatic potential using the Ewald
        summation method."""

        self.p3m_calculator = P3MCalculator(
            potential=CoulombPotential(
                smearing=float(hypers["smearing"]),
                exclusion_radius=neighbor_list_options.cutoff,
            ),
            interpolation_nodes=hypers["interpolation_nodes"],
            full_neighbor_list=neighbor_list_options.full_list,
            mesh_spacing=float(hypers["kspace_resolution"]),
        )
        """Calculator to compute the long-range electrostatic potential using the P3M
        method."""

        self.use_ewald = hypers["use_ewald"]
        """If ``True``, use the Ewald summation method instead of the P3M method for
        periodic systems during training."""

        self.direct_calculator = Calculator(
            potential=CoulombPotential(
                smearing=None,
                exclusion_radius=neighbor_list_options.cutoff,
            ),
            full_neighbor_list=False,  # see docs of torch.combinations
        )
        """Calculator for the electrostatic potential in non-periodic systems."""

        self.neighbor_list_options = neighbor_list_options
        """Neighbor list information for the short-range model."""

        self.norm = torch.nn.RMSNorm(feature_dim)
        """RMS normalization to be applied to the input features."""

        self.charges_map = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, feature_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(feature_dim, self.num_charges),
        )
        """Map the short-range features to atomic charges."""

        self.out_projection = torch.nn.Sequential(
            torch.nn.Linear(self.num_charges, feature_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(feature_dim, feature_dim),
        )
        """Transforms the long-range features before the output."""

    def forward(
        self,
        systems: list[System],
        features: torch.Tensor,
        neighbor_distances: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the long-range features for a list of systems.

        :param systems: A list of :py:class:`System` objects for which to compute the
            long-range features. Each system must contain a neighbor list consistent
            with the neighbor list options used to create the class.
        :param features: A tensor of short-range features for the systems.
        :param neighbor_distances: A tensor of neighbor distances for the systems,
            unused; distances are recomputed from each system neighbor list.
        :return: A tensor of long-range features for the systems.
        """
        features = self.norm(features)
        charges = self.charges_map(features)

        if not torch.jit.is_scripting() and len(systems) > 1:
            potential = self._partitioned_batched_forward(
                systems=systems,
                charges=charges,
            )
            return features + self.out_projection(self.prefactor * potential)

        return features + self.out_projection(
            self.prefactor * self._loop_forward(systems, charges)
        )

    def _partitioned_batched_forward(
        self,
        systems: list[System],
        charges: torch.Tensor,
    ) -> torch.Tensor:
        periodic_systems = []
        periodic_charges = []
        nonperiodic_systems = []
        nonperiodic_charges = []
        output_kinds = []
        output_potentials: list[torch.Tensor] = []

        last_len_nodes = 0
        for system in systems:
            periodic_dimensions = int(system.pbc.sum())
            if periodic_dimensions in (1, 2):
                raise NotImplementedError(
                    "Long-range featurizer does not support systems with only "
                    "one or two periodic dimensions."
                )

            system_num_atoms = len(system)
            system_charges = charges[last_len_nodes : last_len_nodes + system_num_atoms]
            last_len_nodes += system_num_atoms

            if periodic_dimensions == 3 and self.use_ewald and self.training:
                periodic_systems.append(system)
                periodic_charges.append(system_charges)
                output_kinds.append("periodic")
                output_potentials.append(None)
            elif periodic_dimensions == 0:
                nonperiodic_systems.append(system)
                nonperiodic_charges.append(system_charges)
                output_kinds.append("nonperiodic")
                output_potentials.append(None)
            else:
                output_kinds.append("single")
                output_potentials.append(
                    self._single_system_potential(
                        system,
                        system_charges,
                    )
                )

        periodic_outputs: list[torch.Tensor] = []
        if len(periodic_systems) > 0:
            periodic_outputs = list(
                self._batched_ewald_forward(
                    systems=periodic_systems,
                    charges_list=periodic_charges,
                )
            )

        nonperiodic_outputs: list[torch.Tensor] = []
        if len(nonperiodic_systems) > 0:
            nonperiodic_outputs = list(
                self._batched_direct_forward(
                    systems=nonperiodic_systems,
                    charges_list=nonperiodic_charges,
                )
            )

        periodic_index = 0
        nonperiodic_index = 0
        for i, (kind, potential) in enumerate(
            zip(output_kinds, output_potentials, strict=True)
        ):
            if potential is not None:
                continue
            if kind == "periodic":
                output_potentials[i] = periodic_outputs[periodic_index]
                periodic_index += 1
            elif kind == "nonperiodic":
                output_potentials[i] = nonperiodic_outputs[nonperiodic_index]
                nonperiodic_index += 1

        return torch.concatenate(output_potentials)

    def _loop_forward(
        self,
        systems: list[System],
        charges: torch.Tensor,
    ) -> torch.Tensor:
        last_len_nodes = 0
        long_range_features = []
        for system in systems:
            system_charges = charges[last_len_nodes : last_len_nodes + len(system)]
            last_len_nodes += len(system)

            long_range_features.append(
                self._single_system_potential(
                    system,
                    system_charges,
                )
            )

        return torch.concatenate(long_range_features)

    def _single_system_potential(
        self,
        system: System,
        charges: torch.Tensor,
    ) -> torch.Tensor:
        periodic_dimensions = int(system.pbc.sum())
        if periodic_dimensions in (1, 2):
            raise NotImplementedError(
                "Long-range featurizer does not support systems with only "
                "one or two periodic dimensions."
            )

        neighbor_list = system.get_neighbor_list(self.neighbor_list_options)
        neighbor_indices_system = neighbor_list.samples.view(
            ["first_atom", "second_atom"]
        ).values
        neighbor_cell_shifts_system = neighbor_list.samples.values[:, 2:]
        neighbor_distances = torch.sqrt(
            torch.sum(
                (
                    system.positions[neighbor_indices_system[:, 1]]
                    - system.positions[neighbor_indices_system[:, 0]]
                    + neighbor_cell_shifts_system.to(system.cell.dtype) @ system.cell
                )
                ** 2,
                dim=1,
            )
        )

        if periodic_dimensions == 3:
            if self.use_ewald and self.training:
                return self.ewald_calculator.forward(
                    charges=charges,
                    cell=system.cell,
                    positions=system.positions,
                    neighbor_indices=neighbor_indices_system,
                    neighbor_distances=neighbor_distances,
                    periodic=system.pbc,
                )
            return self.p3m_calculator.forward(
                charges=charges,
                cell=system.cell,
                positions=system.positions,
                neighbor_indices=neighbor_indices_system,
                neighbor_distances=neighbor_distances,
                periodic=system.pbc,
            )

        neighbor_indices_system = torch.combinations(
            torch.arange(len(system), device=system.positions.device), 2
        )
        neighbor_distances_system = torch.sqrt(
            torch.sum(
                (
                    system.positions[neighbor_indices_system[:, 1]]
                    - system.positions[neighbor_indices_system[:, 0]]
                )
                ** 2,
                dim=1,
            )
        )
        return self.direct_calculator.forward(
            charges=charges,
            cell=system.cell,
            positions=system.positions,
            neighbor_indices=neighbor_indices_system,
            neighbor_distances=neighbor_distances_system,
        )

    def _batched_ewald_forward(
        self,
        systems: list[System],
        charges_list: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        import torchpme

        i_list = []
        j_list = []
        d_list = []
        pos_list = []
        cell_list = []
        periodic_list = []

        for system in systems:
            neighbor_list = system.get_neighbor_list(self.neighbor_list_options)
            neighbor_indices_system = neighbor_list.samples.view(
                ["first_atom", "second_atom"]
            ).values
            neighbor_cell_shifts_system = neighbor_list.samples.values[:, 2:]
            neighbor_distances_system = torch.sqrt(
                torch.sum(
                    (
                        system.positions[neighbor_indices_system[:, 1]]
                        - system.positions[neighbor_indices_system[:, 0]]
                        + neighbor_cell_shifts_system.to(system.cell.dtype)
                        @ system.cell
                    )
                    ** 2,
                    dim=1,
                )
            )

            i_list.append(neighbor_indices_system[:, 0])
            j_list.append(neighbor_indices_system[:, 1])
            d_list.append(neighbor_distances_system)
            pos_list.append(system.positions)
            cell_list.append(system.cell)
            periodic_list.append(system.pbc)

        atom_counts = torch.tensor(
            [positions.shape[0] for positions in pos_list],
            device=charges_list[0].device,
        )
        pair_counts = torch.tensor(
            [indices.shape[0] for indices in i_list],
            device=charges_list[0].device,
        )

        pos_batch = pad_sequence(pos_list, batch_first=True)
        charges_batch = pad_sequence(charges_list, batch_first=True)
        cell_batch = torch.stack(cell_list)
        periodic_batch = torch.stack(periodic_list)
        i_batch = pad_sequence(i_list, batch_first=True, padding_value=0)
        j_batch = pad_sequence(j_list, batch_first=True, padding_value=0)
        d_batch = pad_sequence(d_list, batch_first=True, padding_value=0.0)

        node_mask = torch.arange(pos_batch.shape[1], device=charges_list[0].device)[
            None, :
        ]
        node_mask = node_mask < atom_counts[:, None]
        pair_mask = torch.arange(i_batch.shape[1], device=charges_list[0].device)[
            None, :
        ]
        pair_mask = pair_mask < pair_counts[:, None]

        kvectors = torchpme.lib.compute_batched_kvectors(
            lr_wavelength=self.ewald_calculator.lr_wavelength,
            cells=cell_batch,
        )

        batched = torch.vmap(self.ewald_calculator.forward)(
            charges_batch,
            cell_batch,
            pos_batch,
            torch.stack((i_batch, j_batch), dim=-1),
            d_batch,
            periodic_batch,
            node_mask,
            pair_mask,
            kvectors,
        )
        return [
            potential[mask] for potential, mask in zip(batched, node_mask, strict=True)
        ]

    def _batched_direct_forward(
        self,
        systems: list[System],
        charges_list: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        pos_list = []
        cell_list = []
        periodic_list = []
        i_list = []
        j_list = []
        d_list = []

        for system in systems:
            neighbor_indices_system = torch.combinations(
                torch.arange(len(system), device=system.positions.device), 2
            )
            neighbor_distances_system = torch.sqrt(
                torch.sum(
                    (
                        system.positions[neighbor_indices_system[:, 1]]
                        - system.positions[neighbor_indices_system[:, 0]]
                    )
                    ** 2,
                    dim=1,
                )
            )
            pos_list.append(system.positions)
            cell_list.append(system.cell)
            periodic_list.append(system.pbc)
            i_list.append(neighbor_indices_system[:, 0])
            j_list.append(neighbor_indices_system[:, 1])
            d_list.append(neighbor_distances_system)

        atom_counts = torch.tensor(
            [positions.shape[0] for positions in pos_list],
            device=charges_list[0].device,
        )
        pair_counts = torch.tensor(
            [indices.shape[0] for indices in i_list],
            device=charges_list[0].device,
        )

        pos_batch = pad_sequence(pos_list, batch_first=True)
        charges_batch = pad_sequence(charges_list, batch_first=True)
        cell_batch = torch.stack(cell_list)
        periodic_batch = torch.stack(periodic_list)
        i_batch = pad_sequence(i_list, batch_first=True, padding_value=0)
        j_batch = pad_sequence(j_list, batch_first=True, padding_value=0)
        d_batch = pad_sequence(d_list, batch_first=True, padding_value=0.0)

        node_mask = torch.arange(pos_batch.shape[1], device=charges_list[0].device)[
            None, :
        ]
        node_mask = node_mask < atom_counts[:, None]
        pair_mask = torch.arange(i_batch.shape[1], device=charges_list[0].device)[
            None, :
        ]
        pair_mask = pair_mask < pair_counts[:, None]

        def wrapped_forward(
            charges: torch.Tensor,
            cell: torch.Tensor,
            positions: torch.Tensor,
            neighbor_indices: torch.Tensor,
            neighbor_distances: torch.Tensor,
            periodic: torch.Tensor,
            node_mask: torch.Tensor,
            pair_mask: torch.Tensor,
        ) -> torch.Tensor:
            return self.direct_calculator.forward(
                charges=charges,
                cell=cell,
                positions=positions,
                neighbor_indices=neighbor_indices,
                neighbor_distances=neighbor_distances,
                periodic=periodic,
                node_mask=node_mask,
                pair_mask=pair_mask,
            )

        batched = torch.vmap(wrapped_forward)(
            charges_batch,
            cell_batch,
            pos_batch,
            torch.stack((i_batch, j_batch), dim=-1),
            d_batch,
            periodic_batch,
            node_mask,
            pair_mask,
        )
        return [
            potential[mask] for potential, mask in zip(batched, node_mask, strict=True)
        ]


class DummyLongRangeFeaturizer(torch.nn.Module):
    # a dummy class for torchscript
    def __init__(self) -> None:
        super().__init__()
        self.use_ewald = True

    def forward(
        self,
        systems: list[System],
        features: torch.Tensor,
        neighbor_distances: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tensor(0)
