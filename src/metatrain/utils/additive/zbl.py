import logging
from typing import Dict, List, Optional

import metatensor.torch
import torch
from ase.data import covalent_radii
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import ModelOutput, NeighborListOptions, System

from ..data import DatasetInfo, TargetInfo
from ..jsonschema import validate
from ..sum_over_atoms import sum_over_atoms


class ZBL(torch.nn.Module):
    """
    A simple model for short-range repulsive interactions.

    The implementation here is equivalent to its
    `LAMMPS counterpart <https://docs.lammps.org/pair_zbl.html>`_, where we set the
    inner cutoff to 0 and the outer cutoff to the sum of the covalent radii of the
    two atoms as tabulated in ASE. Covalent radii that are not available in ASE are
    set to 0.2 Å (and a warning is issued).

    :param model_hypers: A dictionary of model hyperparameters. This contains the
        "inner_cutoff" and "outer_cutoff" keys, which are the inner and outer cutoffs
        for the ZBL potential.
    :param dataset_info: An object containing information about the dataset, including
        target quantities and atomic types.
    """

    def __init__(self, model_hypers: Dict, dataset_info: DatasetInfo):
        super().__init__()

        # `model_hypers` should be an empty dictionary
        validate(
            instance=model_hypers,
            schema={"type": "object", "additionalProperties": False},
        )

        # Check dataset length units
        if dataset_info.length_unit != "angstrom":
            raise ValueError(
                "ZBL only supports angstrom units, but a "
                f"{dataset_info.length_unit} unit was provided."
            )

        for target_name, target_info in dataset_info.targets.items():
            if not self.is_valid_target(target_name, target_info):
                raise ValueError(
                    f"ZBL model does not support target "
                    f"{target_name}. This is an architecture bug. "
                    "Please report this issue and help us improve!"
                )

        self.dataset_info = dataset_info
        self.atomic_types = sorted(dataset_info.atomic_types)

        self.outputs = {
            key: ModelOutput(
                quantity=value.quantity,
                unit=value.unit,
                per_atom=True,
            )
            for key, value in dataset_info.targets.items()
        }

        n_types = len(self.atomic_types)

        self.output_to_output_index = {
            target: i for i, target in enumerate(sorted(dataset_info.targets.keys()))
        }

        self.register_buffer(
            "species_to_index",
            torch.full((max(self.atomic_types) + 1,), -1, dtype=torch.int),
        )
        for i, t in enumerate(self.atomic_types):
            self.species_to_index[t] = i

        self.register_buffer(
            "covalent_radii", torch.empty((n_types,), dtype=torch.float64)
        )
        for i, t in enumerate(self.atomic_types):
            ase_covalent_radius = covalent_radii[t]
            if ase_covalent_radius == 0.2:
                # 0.2 seems to be the default value when the covalent radius
                # is not known/available
                logging.warning(
                    f"Covalent radius for element {t} is not available in ASE. "
                    "Using a default value of 0.2 Å."
                )
            self.covalent_radii[i] = ase_covalent_radius

        largest_covalent_radius = float(torch.max(self.covalent_radii))
        self.cutoff_radius = 2.0 * largest_covalent_radius

    def restart(self, dataset_info: DatasetInfo) -> "ZBL":
        """Restart the model with a new dataset info.

        :param dataset_info: New dataset information to be used.
        """

        for target_name, target_info in dataset_info.targets.items():
            if not self.is_valid_target(target_name, target_info):
                raise ValueError(
                    f"ZBL model does not support target "
                    f"{target_name}. This is an architecture bug. "
                    "Please report this issue and help us improve!"
                )

        return self({}, self.dataset_info.union(dataset_info))

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        """Compute the energies of a system solely based on a ZBL repulsive
        potential.

        :param systems: List of systems to calculate the ZBL energy.
        :param outputs: Dictionary containing the model outputs.
        :param selected_atoms: Optional selection of atoms for which to compute the
            predictions.
        :returns: A dictionary with the computed predictions for each system.

        :raises ValueError: If the `outputs` contain unsupported keys.
        """

        # Assert only one neighbor list for all systems
        neighbor_lists: List[TensorBlock] = []
        for system in systems:
            nl_options = self.requested_neighbor_lists()[0]
            nl = system.get_neighbor_list(nl_options)
            neighbor_lists.append(nl)

        # Find the elements of all i and j atoms
        zi = torch.concatenate(
            [
                system.types[nl.samples.column("first_atom")]
                for nl, system in zip(neighbor_lists, systems)
            ]
        )
        zj = torch.concatenate(
            [
                system.types[nl.samples.column("second_atom")]
                for nl, system in zip(neighbor_lists, systems)
            ]
        )

        # Find the interatomic distances
        rij = torch.concatenate(
            [torch.sqrt(torch.sum(nl.values**2, dim=(1, 2))) for nl in neighbor_lists]
        )

        # Find the ZBL energies
        e_zbl = self.get_pairwise_zbl(zi, zj, rij)

        # Sum over edges to get node energies
        indices_for_sum_list = []
        sum = 0
        for system, nl in zip(systems, neighbor_lists):
            indices_for_sum_list.append(nl.samples.column("first_atom") + sum)
            sum += system.positions.shape[0]

        e_zbl_nodes = torch.zeros(sum, dtype=e_zbl.dtype, device=e_zbl.device)
        e_zbl_nodes.index_add_(0, torch.cat(indices_for_sum_list), e_zbl)

        device = systems[0].positions.device

        # Set the outputs as the ZBL energies
        targets_out: Dict[str, TensorMap] = {}
        for target_key, target in outputs.items():
            sample_values: List[List[int]] = []

            for i_system, system in enumerate(systems):
                sample_values += [[i_system, i_atom] for i_atom in range(len(system))]

            block = TensorBlock(
                values=e_zbl_nodes.reshape(-1, 1),
                samples=Labels(
                    ["system", "atom"], torch.tensor(sample_values, device=device)
                ),
                components=[],
                properties=Labels(
                    names=["energy"], values=torch.tensor([[0]], device=device)
                ),
            )

            targets_out[target_key] = TensorMap(
                keys=Labels(names=["_"], values=torch.tensor([[0]], device=device)),
                blocks=[block],
            )

            # apply selected_atoms to the composition if needed
            if selected_atoms is not None:
                targets_out[target_key] = metatensor.torch.slice(
                    targets_out[target_key], "samples", selected_atoms
                )

            if not target.per_atom:
                targets_out[target_key] = sum_over_atoms(targets_out[target_key])

        return targets_out

    def get_pairwise_zbl(self, zi, zj, rij):
        """
        Ziegler-Biersack-Littmark (ZBL) potential.

        Inputs are the atomic numbers (zi, zj) of the two atoms of interest
        and their distance rij.
        """
        # set cutoff from covalent radii of the elements
        rc = (
            self.covalent_radii[self.species_to_index[zi]]
            + self.covalent_radii[self.species_to_index[zj]]
        )

        r1 = 0.0
        p = 0.23
        # angstrom
        a0 = 0.46850
        c = torch.tensor(
            [0.02817, 0.28022, 0.50986, 0.18175], dtype=rij.dtype, device=rij.device
        )
        d = torch.tensor(
            [0.20162, 0.40290, 0.94229, 3.19980], dtype=rij.dtype, device=rij.device
        )

        a = a0 / (zi**p + zj**p)

        da = d.unsqueeze(-1) / a

        # e * e / (4 * pi * epsilon_0) / electron_volt / angstrom
        factor = 14.399645478425668 * zi * zj
        e = _e_zbl(factor, rij, c, da)  # eV.angstrom

        # switching function
        ec = _e_zbl(factor, rc, c, da)
        dec = _dedr(factor, rc, c, da)
        d2ec = _d2edr2(factor, rc, c, da)

        # coefficients are determined such that E(rc) = 0, E'(rc) = 0, and E''(rc) = 0
        A = (-3 * dec + (rc - r1) * d2ec) / ((rc - r1) ** 2)
        B = (2 * dec - (rc - r1) * d2ec) / ((rc - r1) ** 3)
        C = -ec + (rc - r1) * dec / 2 - (rc - r1) * (rc - r1) * d2ec / 12

        e += A / 3 * ((rij - r1) ** 3) + B / 4 * ((rij - r1) ** 4) + C
        e = e / 2.0  # divide by 2 to fix double counting of edges

        # set all contributions past the cutoff to zero
        e[rij > rc] = 0.0

        return e

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [
            NeighborListOptions(
                cutoff=self.cutoff_radius,
                full_list=True,
                strict=True,
            )
        ]

    @staticmethod
    def is_valid_target(target_name: str, target_info: TargetInfo) -> bool:
        """Finds if a ``TargetInfo`` object is compatible with the ZBL model.

        :param target_info: The ``TargetInfo`` object to be checked.
        """
        if target_info.quantity != "energy":
            logging.debug(
                f"ZBL model does not support target {target_name} since it is "
                "not an energy."
            )
            return False
        if not target_info.is_scalar:
            logging.debug(
                f"ZBL model does not support target {target_name} since it is "
                "not a scalar."
            )
            return False
        if len(target_info.layout.block(0).properties) > 1:
            logging.debug(
                f"ZBL model does not support target {target_name} since it has "
                "more than one property."
            )
            return False
        if target_info.unit != "eV":
            logging.debug(
                f"ZBL model does not support target {target_name} since it is "
                "not in eV."
            )
            return False
        return True


def _phi(r, c, da):
    phi = torch.sum(c.unsqueeze(-1) * torch.exp(-r * da), dim=0)
    return phi


def _dphi(r, c, da):
    dphi = torch.sum(-c.unsqueeze(-1) * da * torch.exp(-r * da), dim=0)
    return dphi


def _d2phi(r, c, da):
    d2phi = torch.sum(c.unsqueeze(-1) * (da**2) * torch.exp(-r * da), dim=0)
    return d2phi


def _e_zbl(factor, r, c, da):
    phi = _phi(r, c, da)
    ret = factor / r * phi
    return ret


def _dedr(factor, r, c, da):
    phi = _phi(r, c, da)
    dphi = _dphi(r, c, da)
    ret = factor / r * (-phi / r + dphi)
    return ret


def _d2edr2(factor, r, c, da):
    phi = _phi(r, c, da)
    dphi = _dphi(r, c, da)
    d2phi = _d2phi(r, c, da)

    ret = factor / r * (d2phi - 2 / r * dphi + 2 * phi / (r**2))
    return ret
