from math import factorial

import numpy as np
import torch
from ase.data import covalent_radii

from .adaptive_cutoff import get_adaptive_cutoffs
from .cutoff_function import cutoff_func
from .physical_basis import get_physical_basis_spliner


class Precomputer(torch.nn.Module):
    def __init__(
        self,
        max_eigenvalue,
        cutoff,
        cutoff_width,
        scale,
        optimizable_lengthscales,
        all_species,
        use_sphericart,
        num_neighbors_adaptive=16,
    ):
        super().__init__()

        self.n_max_l, self.spliner = get_physical_basis_spliner(
            max_eigenvalue, cutoff, normalize=True
        )
        self.l_max = len(self.n_max_l) - 1

        self.spherical_harmonics_split_list = [
            (2 * l + 1)
            for l in range(self.l_max + 1)  # noqa: E741
        ]
        if use_sphericart:
            self.spherical_harmonics = SphericalHarmonicsSphericart(self.l_max)
        else:
            self.spherical_harmonics = SphericalHarmonicsNoSphericart(self.l_max)

        lengthscales = torch.zeros((max(all_species) + 1))
        for species in all_species:
            lengthscales[species] = np.log(scale * covalent_radii[species])

        if optimizable_lengthscales:
            self.lengthscales = torch.nn.Parameter(lengthscales)
        else:
            self.register_buffer("lengthscales", lengthscales)

        self.num_neighbors_adaptive = float(num_neighbors_adaptive)
        self.cutoff_width = float(cutoff_width)
        self.r_cut = float(cutoff)

    def forward(
        self,
        positions,
        cells,
        cell_shifts,
        center_indices,
        neighbor_indices,
        structure_pairs,
        center_species,
        neighbor_species,
    ):
        cartesian_vectors = get_cartesian_vectors(
            positions,
            cells,
            cell_shifts,
            center_indices,
            neighbor_indices,
            structure_pairs,
        )

        r = torch.sqrt((cartesian_vectors**2).sum(dim=-1))

        ##### 1. Adaptive cutoff #####
        if self.num_neighbors_adaptive is not None:
            # Adaptive cutoff scheme to approximately select `num_neighbors_adaptive`
            # neighbors for each atom
            atomic_cutoffs = get_adaptive_cutoffs(
                center_indices,
                r,
                self.num_neighbors_adaptive,
                len(positions),
                self.r_cut,
                cutoff_width=self.cutoff_width,
            )
            # Symmetrize the cutoffs between pairs of atoms (PET needs this symmetry
            # due to its corresponding edge indexing ij -> ji)
            pair_cutoffs = (
                atomic_cutoffs[center_indices] + atomic_cutoffs[neighbor_indices]
            ) / 2.0
            # Apply cutoff mask
            cutoff_mask = r <= pair_cutoffs
            pair_cutoffs = pair_cutoffs[cutoff_mask]
            center_indices = center_indices[cutoff_mask]
            neighbor_indices = neighbor_indices[cutoff_mask]
            cartesian_vectors = cartesian_vectors[cutoff_mask]
            cell_shifts = cell_shifts[cutoff_mask]
            r = r[cutoff_mask]
            center_species = center_species[cutoff_mask]
            neighbor_species = neighbor_species[cutoff_mask]
        else:
            pair_cutoffs = self.r_cut * torch.ones_like(r)

        ##### 2. Spherical harmonics #####
        spherical_harmonics = self.spherical_harmonics(
            cartesian_vectors
        )  # Get the spherical harmonics
        spherical_harmonics = spherical_harmonics * (4.0 * torch.pi) ** (
            0.5
        )  # normalize them
        spherical_harmonics = torch.split(
            spherical_harmonics, self.spherical_harmonics_split_list, dim=1
        )  # Split them into l chunks

        ##### 3. Radial basis (including cutoff function) #####
        x = r / (
            0.1
            + torch.exp(self.lengthscales[center_species])
            + torch.exp(self.lengthscales[neighbor_species])
        )
        capped_x = torch.where(x < 10.0, x, 5.0)
        radial_functions = torch.where(
            x.unsqueeze(1) < 10.0, self.spliner.compute(capped_x), 0.0
        )
        cutoff_multiplier = cutoff_func(r, pair_cutoffs, self.cutoff_width)
        radial_functions = radial_functions * cutoff_multiplier.unsqueeze(1)
        radial_basis = torch.split(radial_functions, self.n_max_l, dim=1)

        return (
            center_indices,
            neighbor_indices,
            center_species,
            neighbor_species,
            spherical_harmonics,
            radial_basis,
        )


def get_cartesian_vectors(
    positions, cells, cell_shifts, center_indices, neighbor_indices, structure_pairs
):
    """
    Calculate direction vectors between center and neighbor atoms.

    :param positions: Atomic positions [N_total, 3]
    :param cells: Unit cells [N_structures, 3, 3]
    :param cell_shifts: Cell shift vectors [N_pairs, 3]
    :param center_indices: Global center indices [N_pairs]
    :param neighbor_indices: Global neighbor indices [N_pairs]
    :param structure_pairs: Structure index for each pair [N_pairs]
    :return: Direction vectors from center to neighbor [N_pairs, 3]
    """
    direction_vectors = (
        positions[neighbor_indices]
        - positions[center_indices]
        + torch.einsum(
            "ab, abc -> ac", cell_shifts.to(cells.dtype), cells[structure_pairs]
        )
    )
    return direction_vectors


class SphericalHarmonicsSphericart(torch.nn.Module):
    def __init__(self, l_max):
        super(SphericalHarmonicsSphericart, self).__init__()
        # import sphericart here conditionally otherwise it will be registered as
        # an extension even if we don't use it
        import sphericart.torch

        self.spherical_harmonics_calculator = sphericart.torch.SphericalHarmonics(
            l_max, normalized=True
        )

    def forward(self, xyz):
        return self.spherical_harmonics_calculator.compute(xyz)


class SphericalHarmonicsNoSphericart(torch.nn.Module):
    # uses the sphericart algorithm implemented in pytorch
    def __init__(self, l_max):
        super(SphericalHarmonicsNoSphericart, self).__init__()
        self.l_max = l_max

        self.register_buffer(
            "F", torch.empty(((self.l_max + 1) * (self.l_max + 2) // 2,))
        )
        for l in range(l_max + 1):  # noqa: E741
            for m in range(0, l + 1):
                self.F[l * (l + 1) // 2 + m] = (-1) ** m * np.sqrt(
                    (2 * l + 1) / (2 * np.pi) * factorial(l - m) / factorial(l + m)
                )

    def forward(self, xyz):
        device = xyz.device
        dtype = xyz.dtype

        rsq = torch.sum(xyz**2, dim=1)
        xyz = xyz / torch.sqrt(rsq).unsqueeze(1)

        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        Q = torch.empty(
            (xyz.shape[0], (self.l_max + 1) * (self.l_max + 2) // 2),
            device=device,
            dtype=dtype,
        )
        Q[:, 0] = 1.0
        for l in range(1, self.l_max + 1):  # noqa: E741
            Q[:, (l + 1) * (l + 2) // 2 - 1] = (
                -(2 * l - 1) * Q[:, l * (l + 1) // 2 - 1].clone()
            )
            Q[:, (l + 1) * (l + 2) // 2 - 2] = (
                -z * Q[:, (l + 1) * (l + 2) // 2 - 1].clone()
            )
            for m in range(0, l - 1):
                Q[:, l * (l + 1) // 2 + m] = (
                    (2 * l - 1) * z * Q[:, (l - 1) * l // 2 + m].clone()
                    - (l + m - 1) * Q[:, (l - 2) * (l - 1) // 2 + m].clone()
                ) / (l - m)

        s = torch.empty((xyz.shape[0], self.l_max + 1), device=device, dtype=dtype)
        c = torch.empty((xyz.shape[0], self.l_max + 1), device=device, dtype=dtype)

        s[:, 0] = 0.0
        c[:, 0] = 1.0
        for m in range(1, self.l_max + 1):
            s[:, m] = x * s[:, m - 1].clone() + y * c[:, m - 1].clone()
            c[:, m] = x * c[:, m - 1].clone() - y * s[:, m - 1].clone()

        Y = torch.empty(
            (xyz.shape[0], (self.l_max + 1) * (self.l_max + 1)),
            device=device,
            dtype=dtype,
        )
        for l in range(self.l_max + 1):  # noqa: E741
            for m in range(-l, 0):
                Y[:, l * l + l + m] = (
                    self.F[l * (l + 1) // 2 - m] * Q[:, l * (l + 1) // 2 - m] * s[:, -m]
                )
            Y[:, l * l + l] = (
                self.F[l * (l + 1) // 2]
                * Q[:, l * (l + 1) // 2]
                / torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype))
            )
            for m in range(1, l + 1):
                Y[:, l * l + l + m] = (
                    self.F[l * (l + 1) // 2 + m] * Q[:, l * (l + 1) // 2 + m] * c[:, m]
                )

        return Y
