"""Modules to allow SOAP-BPNN to fit arbitrary spherical tensor targets."""

import copy
from typing import Any, Dict, List, Optional, Tuple

import metatensor.torch as mts
import numpy as np
import sphericart.torch
import torch
import wigners
from metatensor.torch import Labels, TensorMap, TensorBlock
from metatensor.torch.learn.nn import Linear as LinearMap
from spex import SphericalExpansion
from torch.profiler import record_function


class VectorBasis(torch.nn.Module):
    """
    This module creates a basis of 3 vectors for each atomic environment.

    In practice, this is done by contracting a l=1 spherical expansion.

    :param atomic_types: list of atomic types in the dataset.
    :param soap_hypers: dictionary with the SOAP hyper-parameters.
    """

    def __init__(self, atomic_types: List[int], soap_hypers: Dict[str, Any]) -> None:
        super().__init__()
        self.atomic_types = atomic_types
        # Define a new hyper-parameter for the basis part of the expansion
        soap_hypers = copy.deepcopy(soap_hypers)

        spex_soap_hypers = {
            "cutoff": soap_hypers["cutoff"]["radius"],
            "max_angular": 1,
            "radial": {
                "LaplacianEigenstates": {
                    "max_radial": soap_hypers["max_radial"],
                }
            },
            "angular": "SphericalHarmonics",
            "cutoff_function": {
                "ShiftedCosine": {"width": soap_hypers["cutoff"]["width"]}
            },
            "species": {"Orthogonal": {"species": self.atomic_types}},
        }

        self.soap_calculator = SphericalExpansion(**spex_soap_hypers)

        self.neighbor_species_labels = Labels(
            names=["neighbor_type"],
            values=torch.tensor(self.atomic_types).reshape(-1, 1),
        )

        self.contraction = LinearMap(
            in_keys=Labels(
                names=["o3_lambda", "o3_sigma", "center_type"],
                values=torch.stack(
                    [
                        torch.tensor([1] * len(self.atomic_types)),
                        torch.tensor([1] * len(self.atomic_types)),
                        torch.tensor(self.atomic_types),
                    ],
                    dim=1,
                ),
            ),
            in_features=(self.soap_calculator.radial.n_per_l[1])
            * len(self.atomic_types),
            out_features=3,
            bias=False,
            out_properties=[Labels.range("basis", 3) for _ in self.atomic_types],
        )
        # this optimizable basis seems to work much better than a fixed one

    def forward(
        self,
        interatomic_vectors: torch.Tensor,
        centers: torch.Tensor,
        neighbors: torch.Tensor,
        species: torch.Tensor,
        structures: torch.Tensor,
        atom_index_in_structure: torch.Tensor,
        selected_atoms: Optional[Labels],
    ) -> torch.Tensor:
        """
        Compute a basis of 3 vectors for each atomic environment.

        :param interatomic_vectors: (num_edges, 3) tensor with the vectors from
            each center to each neighbor.
        :param centers: (num_edges,) tensor with the indices of the center atoms.
        :param neighbors: (num_edges,) tensor with the indices of the neighbor
            atoms.
        :param species: (num_atoms,) tensor with the atomic species of each atom.
        :param structures: (num_atoms,) tensor with the index of the structure each
            atom belongs to.
        :param atom_index_in_structure: (num_atoms,) tensor with the index of each atom
            in its structure.
        :param selected_atoms: optional Labels object to select a subset of the atoms
            to compute the basis for.
        :return: a tensor of shape (num_atoms, 3, 3) with the basis of 3 vectors for
            each atomic environment
        """
        device = interatomic_vectors.device

        if self.neighbor_species_labels.device != device:
            self.neighbor_species_labels = self.neighbor_species_labels.to(device)

        l1_spherical_expansion = self.soap_calculator(
            interatomic_vectors,
            centers,
            neighbors,
            species,
        )[1]  # only l=1 tensor

        l1_spherical_expansion = l1_spherical_expansion.reshape(
            l1_spherical_expansion.shape[0],
            l1_spherical_expansion.shape[1],
            l1_spherical_expansion.shape[2] * l1_spherical_expansion.shape[3],
        )  # [center, o3_mu, features]

        with record_function("conversion"):

            unique_center_species = torch.unique(species)
            blocks: list[TensorBlock] = []
            for s in unique_center_species:
                mask = species == s
                l1_spherical_expansion_filtered = l1_spherical_expansion[mask]
                structures_filtered = structures[mask]
                centers_filtered = atom_index_in_structure[mask]
                block = TensorBlock(
                    values=l1_spherical_expansion_filtered,
                    samples=Labels(
                        names=["system", "atom"],
                        values=torch.stack(
                            [structures_filtered, centers_filtered], dim=1
                        ),
                    ),
                    components=[
                        Labels(
                            names=["o3_mu"],
                            values=torch.tensor(
                                [-1, 0, 1], dtype=torch.long, device=device
                            ).unsqueeze(1),
                        )
                    ],
                    properties=Labels(
                        names=["property"],
                        values=torch.arange(l1_spherical_expansion.shape[2], device=l1_spherical_expansion.device).unsqueeze(1),
                    )
                )
                blocks.append(block)
            l1_spherical_expansion_as_tensor_map = TensorMap(
                keys=Labels(
                    names=["o3_lambda", "o3_sigma", "center_type"],
                    values=torch.tensor([[1, 1, int(s)] for s in unique_center_species], device=device),
                ),
                blocks=blocks,
            )

        if selected_atoms is not None:
            l1_spherical_expansion_as_tensor_map = mts.slice(
                l1_spherical_expansion_as_tensor_map, "samples", selected_atoms
            )

        with record_function("contraction"):
            basis_vectors = self.contraction(l1_spherical_expansion_as_tensor_map)

        # The following (until the end of the function) is equivalent to
        # basis_vectors_as_tensor = (
        #     basis_vectors.keys_to_samples("center_type")
        # ).block()
        # but faster

        all_basis_vectors = torch.concatenate(
            [b.values for b in basis_vectors.blocks()]
        )
        # however, we need to sort them according to the order of the
        # atoms in the systems
        system_sizes = torch.bincount(structures, minlength=len(torch.unique(structures)))
        system_offsets = torch.cat(
            [
                torch.tensor([0], device=device),
                torch.cumsum(system_sizes, dim=0)[:-1],
            ]
        )
        all_system_indices = torch.concatenate(
            [
                b.samples.values[:, 0]
                for b in basis_vectors.blocks()
            ]
        )
        all_atom_indices = torch.concatenate(
            [
                b.samples.values[:, 1]
                for b in basis_vectors.blocks()
            ]
        )
        overall_atom_indices = (
            system_offsets[all_system_indices] + all_atom_indices
        )
        sorting_indices = torch.argsort(overall_atom_indices)
        basis_vectors_as_tensor = all_basis_vectors[
            sorting_indices
        ]
        return basis_vectors_as_tensor  # [n_atoms, 3(yzx), 3]


class TensorBasis(torch.nn.Module):
    """
    Creates a basis of spherical tensors for each atomic environment. Internally, it
    uses one (for proper tensors) or two (for pseudotensors) VectorBasis objects to
    build a basis of 3 vectors.

    :param atomic_types: list of atomic types in the dataset.
    :param soap_hypers: dictionary with the SOAP hyper-parameters.
    :param o3_lambda: the integer order of the spherical tensor to be built.
    :param o3_sigma: 1 for proper tensors, -1 for pseudotensors.
    :param add_lambda_basis: if True and o3_lambda>1, adds a contracted
        lambda-basis to the spherical tensor basis. This is done by contracting a
        spherical expansion with the same o3_lambda as the target tensor.
        This usually improves the performance of the model.
    """

    cgs: Dict[str, torch.Tensor]  # torchscript needs this
    """ dictionary with the Clebsch-Gordan coefficients"""

    def __init__(
        self,
        atomic_types: List[int],
        soap_hypers: Dict[str, Any],
        o3_lambda: int,
        o3_sigma: int,
        add_lambda_basis: bool,
    ) -> None:
        super().__init__()

        self.o3_lambda = o3_lambda
        self.o3_sigma = o3_sigma
        if self.o3_lambda > 0:
            self.vector_basis = VectorBasis(atomic_types, soap_hypers)
        else:
            self.vector_basis = FakeVectorBasis()  # needed to make torchscript work
        if self.o3_sigma == -1:
            self.vector_basis_pseudotensor = VectorBasis(atomic_types, soap_hypers)
        else:
            self.vector_basis_pseudotensor = FakeVectorBasis()  # make torchscript work

        if self.o3_lambda > 1:
            self.spherical_harmonics_calculator = sphericart.torch.SolidHarmonics(
                l_max=self.o3_lambda
            )
        else:
            # needed to make torchscript work
            self.spherical_harmonics_calculator = torch.nn.Identity()

        if self.o3_lambda > 1 or self.o3_sigma == -1:
            self.cgs = {
                f"{l1}_{l2}_{L}": cg_tensor
                for (l1, l2, L), cg_tensor in get_cg_coefficients(
                    max(self.o3_lambda, 1)  # need at least 1 for pseudoscalar case
                )._cgs.items()
            }
        else:
            # needed to make torchscript work
            self.cgs = {}  # type: ignore

        self.atomic_types = atomic_types
        self.neighbor_species_labels = Labels(
            names=["neighbor_type"],
            values=torch.tensor(self.atomic_types).reshape(-1, 1),
        )
        if add_lambda_basis and o3_lambda > 1:
            soap_hypers = copy.deepcopy(soap_hypers)
            spex_soap_hypers = {
                "cutoff": soap_hypers["cutoff"]["radius"],
                "max_angular": o3_lambda,
                "radial": {
                    "LaplacianEigenstates": {
                        "max_radial": soap_hypers["max_radial"],
                    }
                },
                "angular": "SphericalHarmonics",
                "cutoff_function": {
                    "ShiftedCosine": {"width": soap_hypers["cutoff"]["width"]}
                },
                "species": {"Orthogonal": {"species": self.atomic_types}},
            }
            self.spex_calculator = SphericalExpansion(**spex_soap_hypers)
            self.spex_contraction = LinearMap(
                in_keys=Labels(
                    names=["o3_lambda", "o3_sigma", "center_type"],
                    values=torch.stack(
                        [
                            torch.tensor([o3_lambda] * len(self.atomic_types)),
                            torch.tensor([1] * len(self.atomic_types)),
                            torch.tensor(self.atomic_types),
                        ],
                        dim=1,
                    ),
                ),
                in_features=(self.spex_calculator.radial.n_per_l[o3_lambda])
                * len(self.atomic_types),
                out_features=2 * o3_lambda + 1,
                bias=False,
                out_properties=[
                    Labels.range("lambda_basis", 2 * o3_lambda + 1)
                    for _ in self.atomic_types
                ],
            )
        else:
            # needed to make torchscript work
            self.spex_calculator = FakeSphericalExpansion()
            self.spex_contraction = FakeLinearMap()
        self.add_lambda_basis = add_lambda_basis and o3_lambda > 1

    def forward(
        self,
        interatomic_vectors: torch.Tensor,
        centers: torch.Tensor,
        neighbors: torch.Tensor,
        species: torch.Tensor,
        structures: torch.Tensor,
        atom_index_in_structure: torch.Tensor,
        selected_atoms: Optional[Labels],
    ) -> torch.Tensor:
        """
        Compute the basis of spherical tensors for each atomic environment.

        :param interatomic_vectors: (num_edges, 3) tensor with the vectors from
            each center to each neighbor.
        :param centers: (num_edges,) tensor with the indices of the center atoms.
        :param neighbors: (num_edges,) tensor with the indices of the neighbor
            atoms.
        :param species: (num_atoms,) tensor with the atomic species of each atom.
        :param structures: (num_atoms,) tensor with the index of the structure each
            atom belongs to.
        :param atom_index_in_structure: (num_atoms,) tensor with the index of each atom
            in its structure.
        :param selected_atoms: optional Labels object to select a subset of the atoms
            to compute the basis for.
        :return: a tensor of shape (num_atoms, 2*o3_lambda+1, 2*o3_lambda+1) with the
            basis of spherical tensors for each atomic environment.
        If add_lambda_basis is True, the shape is
        (num_atoms, 2*o3_lambda+1, 2*o3_lambda+1 + 2*o3_lambda+1)
        and the last 2*o3_lambda+1 components correspond to the contracted
        lambda basis.
        """

        # transfer cg dict to device and dtype if needed
        device = interatomic_vectors.device
        dtype = interatomic_vectors.dtype
        for k, v in self.cgs.items():
            if v.device != device or v.dtype != dtype:
                self.cgs[k] = v.to(device, dtype)

        if selected_atoms is None:
            num_atoms = len(atom_index_in_structure)
        else:
            num_atoms = len(selected_atoms)

        if self.o3_lambda == 0:
            basis = torch.ones(
                (num_atoms, 1, 1),
                device=device,
                dtype=dtype,
            )
        elif self.o3_lambda == 1:
            with record_function("vector_basis"):
                basis = self.vector_basis(
                    interatomic_vectors,
                    centers,
                    neighbors,
                    species,
                    structures,
                    atom_index_in_structure,
                    selected_atoms,
                )
        elif self.o3_lambda == 2:
            basis = torch.empty(
                (num_atoms, 5, 5),
                device=device,
                dtype=dtype,
            )
            vector_basis = self.vector_basis(
                interatomic_vectors,
                centers,
                neighbors,
                species,
                structures,
                atom_index_in_structure,
                selected_atoms,
            )
            # vector_basis is [n_atoms, 3(yzx), 3]
            vector_1_xyz = vector_basis[:, [2, 0, 1], 0]
            vector_2_xyz = vector_basis[:, [2, 0, 1], 1]
            basis[:, :, 0] = self.spherical_harmonics_calculator(vector_1_xyz)[:, 4:]
            basis[:, :, 1] = self.spherical_harmonics_calculator(vector_2_xyz)[:, 4:]
            vector_1_spherical = vector_basis[:, :, 0]
            vector_2_spherical = vector_basis[:, :, 1]
            vector_3_spherical = vector_basis[:, :, 2]
            basis[:, :, 2] = cg_combine(
                vector_1_spherical, vector_2_spherical, self.cgs["1_1_2"]
            )
            basis[:, :, 3] = cg_combine(
                vector_1_spherical, vector_3_spherical, self.cgs["1_1_2"]
            )
            basis[:, :, 4] = cg_combine(
                vector_2_spherical, vector_3_spherical, self.cgs["1_1_2"]
            )
        else:  # self.o3_lambda > 2
            basis = torch.empty(
                (
                    num_atoms,
                    2 * self.o3_lambda + 1,
                    2 * self.o3_lambda + 1,
                ),
                device=device,
                dtype=dtype,
            )
            vector_basis = self.vector_basis(
                interatomic_vectors,
                centers,
                neighbors,
                species,
                structures,
                atom_index_in_structure,
                selected_atoms,
            )
            # vector_basis is [n_atoms, 3(yzx), 3]
            vector_1_xyz = vector_basis[:, [2, 0, 1], 0]
            vector_2_xyz = vector_basis[:, [2, 0, 1], 1]
            vector_3_spherical = vector_basis[:, :, 2]
            sh_1 = self.spherical_harmonics_calculator(vector_1_xyz)
            sh_2 = self.spherical_harmonics_calculator(vector_2_xyz)
            for lam in range(self.o3_lambda + 1):
                basis[:, :, lam] = cg_combine(
                    sh_1[:, lam * lam : (lam + 1) * (lam + 1)],
                    sh_2[
                        :,
                        (self.o3_lambda - lam) * (self.o3_lambda - lam) : (
                            (self.o3_lambda - lam) + 1
                        )
                        * ((self.o3_lambda - lam) + 1),
                    ],
                    self.cgs[
                        str(lam)
                        + "_"
                        + str(self.o3_lambda - lam)
                        + "_"
                        + str(self.o3_lambda)
                    ],
                )
            for lam in range(self.o3_lambda):
                basis[:, :, self.o3_lambda + 1 + lam] = cg_combine(
                    cg_combine(
                        sh_1[:, lam * lam : (lam + 1) * (lam + 1)],
                        sh_2[
                            :,
                            (self.o3_lambda - lam - 1) * (self.o3_lambda - lam - 1) : (
                                (self.o3_lambda - lam - 1) + 1
                            )
                            * ((self.o3_lambda - lam - 1) + 1),
                        ],
                        self.cgs[
                            str(lam)
                            + "_"
                            + str(self.o3_lambda - lam - 1)
                            + "_"
                            + str(self.o3_lambda - 1)
                        ],
                    ),
                    vector_3_spherical,
                    self.cgs[str(self.o3_lambda - 1) + "_1_" + str(self.o3_lambda)],
                )

        if self.add_lambda_basis:
            # add the lambda basis
            lambda_basis = self.spex_calculator(
                interatomic_vectors,
                centers,
                neighbors,
                species,
                structures,
                atom_index_in_structure,
            )
            if selected_atoms is not None:
                lambda_basis = mts.slice(lambda_basis, "samples", selected_atoms)
            lambda_basis = lambda_basis.keys_to_properties(self.neighbor_species_labels)
            lambda_basis = mts.drop_blocks(
                lambda_basis,
                keys=Labels(
                    ["o3_lambda", "o3_sigma"],
                    torch.tensor(
                        [[ell, 1] for ell in range(self.o3_lambda)], device=device
                    ),
                ),
            )

            lambda_basis = self.spex_contraction(lambda_basis)
            lambda_basis = lambda_basis.keys_to_samples("center_type")
            basis = torch.cat(
                (
                    basis,
                    lambda_basis.block({"o3_lambda": self.o3_lambda}).values,
                ),
                dim=-1,
            )

        if self.o3_sigma == -1:
            # multiply by pseudoscalar
            vector_basis_pseudotensor = self.vector_basis_pseudotensor(
                interatomic_vectors,
                centers,
                neighbors,
                species,
                structures,
                atom_index_in_structure,
                selected_atoms,
            )
            vector_1_spherical = vector_basis_pseudotensor[:, :, 0]
            vector_2_spherical = vector_basis_pseudotensor[:, :, 1]
            vector_3_spherical = vector_basis_pseudotensor[:, :, 2]
            pseudoscalar = cg_combine(
                cg_combine(vector_1_spherical, vector_2_spherical, self.cgs["1_1_1"]),
                vector_3_spherical,
                self.cgs["1_1_0"],
            )
            basis = basis * pseudoscalar.unsqueeze(1)

        return basis  # [n_atoms, 2*o3_lambda+1, 2*o3_lambda+1]


def cg_combine(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """
    Combine two spherical tensors A and B using the Clebsch-Gordan coefficients C.

    :param A: Tensor of shape (n, 2*l1+1)
    :param B: Tensor of shape (n, 2*l2+1)
    :param C: Tensor of shape (2*l1+1, 2*l2+1, 2*L+1)
    :return: Tensor of shape (n, 2*L+1)
    """
    return torch.einsum("im, in, mnp-> ip", A, B, C)


def get_cg_coefficients(l_max: int) -> "ClebschGordanReal":
    """
    Get the Clebsch-Gordan coefficients for all combinations of l1, l2, L up to l_max.

    :param l_max: Maximum value of l1, l2, L.
    :return: A ClebschGordanReal object containing the coefficients.
    """
    cg_object = ClebschGordanReal()
    for l1 in range(l_max + 1):
        for l2 in range(l_max + 1):
            for L in range(abs(l1 - l2), min(l1 + l2, l_max) + 1):
                cg_object._add(l1, l2, L)
    return cg_object


class ClebschGordanReal:
    def __init__(self) -> None:
        self._cgs: Dict[Tuple[int, int, int], torch.Tensor] = {}

    def _add(self, l1: int, l2: int, L: int) -> None:
        if self._cgs is None:
            raise ValueError("Trying to add CGs when not initialized... exiting")

        if (l1, l2, L) in self._cgs:
            raise ValueError("Trying to add CGs that are already present... exiting")

        maxx = max(l1, max(l2, L))

        # real-to-complex and complex-to-real transformations as matrices
        r2c = {}
        c2r = {}
        for l in range(0, maxx + 1):  # noqa: E741
            r2c[l] = _real2complex(l)
            c2r[l] = np.conjugate(r2c[l]).T

        complex_cg = _complex_clebsch_gordan_matrix(l1, l2, L)

        real_cg = (r2c[l1].T @ complex_cg.reshape(2 * l1 + 1, -1)).reshape(
            complex_cg.shape
        )

        real_cg = real_cg.swapaxes(0, 1)
        real_cg = (r2c[l2].T @ real_cg.reshape(2 * l2 + 1, -1)).reshape(real_cg.shape)
        real_cg = real_cg.swapaxes(0, 1)

        real_cg = real_cg @ c2r[L].T

        if (l1 + l2 + L) % 2 == 0:
            rcg = np.real(real_cg)
        else:
            rcg = np.imag(real_cg)

        # Zero any possible (and very rare) near-zero elements
        where_almost_zero = np.where(
            np.logical_and(np.abs(rcg) > 0, np.abs(rcg) < 1e-14)
        )
        if len(where_almost_zero[0] != 0):
            print("INFO: Found almost-zero CG!")
        for i0, i1, i2 in zip(
            where_almost_zero[0],
            where_almost_zero[1],
            where_almost_zero[2],
            strict=True,
        ):
            rcg[i0, i1, i2] = 0.0

        self._cgs[(l1, l2, L)] = torch.tensor(rcg)

    def get(self, key: Tuple[int, int, int]) -> torch.Tensor:
        if key in self._cgs:
            return self._cgs[key]
        else:
            self._add(key[0], key[1], key[2])
            return self._cgs[key]


def _real2complex(L: int) -> np.ndarray:
    """
    Computes a matrix that can be used to convert from real to complex-valued
    spherical harmonics(coefficients) of order L.

    It's meant to be applied to the left, ``real2complex @ [-L..L]``.

    :param L: the order of the spherical harmonics.
    :return: a (2L+1, 2L+1) matrix that converts real to complex spherical harmonics.
    """
    result = np.zeros((2 * L + 1, 2 * L + 1), dtype=np.complex128)

    I_SQRT_2 = 1.0 / np.sqrt(2)

    for m in range(-L, L + 1):
        if m < 0:
            result[L - m, L + m] = I_SQRT_2 * 1j * (-1) ** m
            result[L + m, L + m] = -I_SQRT_2 * 1j

        if m == 0:
            result[L, L] = 1.0

        if m > 0:
            result[L + m, L + m] = I_SQRT_2 * (-1) ** m
            result[L - m, L + m] = I_SQRT_2

    return result


def _complex_clebsch_gordan_matrix(l1: int, l2: int, L: int) -> np.ndarray:
    if np.abs(l1 - l2) > L or np.abs(l1 + l2) < L:
        return np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * L + 1), dtype=np.double)
    else:
        return wigners.clebsch_gordan_array(l1, l2, L)


class FakeVectorBasis(torch.nn.Module):
    # fake class to make torchscript work

    def forward(
        self,
        interatomic_vectors: torch.Tensor,
        centers: torch.Tensor,
        neighbors: torch.Tensor,
        species: torch.Tensor,
        structures: torch.Tensor,
        atom_index_in_structure: torch.Tensor,
        selected_atoms: Optional[Labels],
    ) -> torch.Tensor:
        return torch.tensor(0)


class FakeSphericalExpansion(torch.nn.Module):
    # Dummy class to make torchscript work
    def forward(
        self,
        interatomic_vectors: torch.Tensor,
        centers: torch.Tensor,
        neighbors: torch.Tensor,
        species: torch.Tensor,
        structures: torch.Tensor,
        atom_index_in_structure: torch.Tensor,
    ) -> TensorMap:
        return TensorMap(
            keys=Labels(names=["dummy"], values=torch.tensor([[]])), blocks=[]
        )


class FakeLinearMap(torch.nn.Module):
    # fake class to make torchscript work

    def forward(
        self,
        tensor_map: TensorMap,
    ) -> TensorMap:
        return tensor_map
