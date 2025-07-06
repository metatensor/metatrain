"""Modules to allow SOAP-BPNN to fit arbitrary spherical tensor targets."""

import copy
from typing import Dict, Optional

import metatensor.torch
import numpy as np
import sphericart.torch
import torch
import wigners
from metatensor.torch import Labels, TensorMap
from metatensor.torch.learn.nn import Linear as LinearMap
from spex.metatensor import SphericalExpansion


class VectorBasis(torch.nn.Module):
    """
    This module creates a basis of 3 vectors for each atomic environment.

    In practice, this is done by contracting a l=1 spherical expansion.
    """

    def __init__(self, atomic_types, soap_hypers) -> None:
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
            in_features=(self.soap_calculator.calculator.radial.n_per_l[1])
            * len(self.atomic_types),
            out_features=3,
            bias=False,
            out_properties=[Labels.range("basis", 3) for _ in self.atomic_types],
        )
        # this optimizable basis seems to work much better than a fixed one

    def forward(
        self,
        interatomic_vectors,
        centers,
        neighbors,
        species,
        structures,
        atom_index_in_structure,
        selected_atoms: Optional[Labels],
    ) -> torch.Tensor:
        device = interatomic_vectors.device

        if self.neighbor_species_labels.device != device:
            self.neighbor_species_labels = self.neighbor_species_labels.to(device)

        spherical_expansion = self.soap_calculator(
            interatomic_vectors,
            centers,
            neighbors,
            species,
            structures,
            atom_index_in_structure,
        )
        if selected_atoms is not None:
            spherical_expansion = metatensor.torch.slice(
                spherical_expansion, "samples", selected_atoms
            )

        # by calling keys_to_samples and keys_to_properties in the same order as they
        # are called in the main model, we should ensure that the order of the samples
        # is the same
        spherical_expansion = spherical_expansion.keys_to_properties(
            self.neighbor_species_labels
        )

        # drop all L=0 blocks
        spherical_expansion = metatensor.torch.drop_blocks(
            spherical_expansion,
            keys=Labels(
                ["o3_lambda", "o3_sigma"], torch.tensor([[0, 1]], device=device)
            ),
        )

        basis_vectors = self.contraction(spherical_expansion)
        basis_vectors = basis_vectors.keys_to_samples("center_type")

        basis_vectors_as_tensor = basis_vectors.block({"o3_lambda": 1}).values
        return basis_vectors_as_tensor  # [n_atoms, 3(yzx), 3]


class TensorBasis(torch.nn.Module):
    """
    Creates a basis of spherical tensors for each atomic environment. Internally, it
    uses one (for proper tensors) or two (for pseudotensors) VectorBasis objects to
    build a basis of 3 vectors.
    """

    cgs: Dict[str, torch.Tensor]  # torchscript needs this

    def __init__(
        self, atomic_types, soap_hypers, o3_lambda, o3_sigma, add_lambda_basis
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
                in_features=(self.spex_calculator.calculator.radial.n_per_l[o3_lambda])
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
        interatomic_vectors,
        centers,
        neighbors,
        species,
        structures,
        atom_index_in_structure,
        selected_atoms: Optional[Labels],
    ) -> torch.Tensor:
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
                lambda_basis = metatensor.torch.slice(
                    lambda_basis, "samples", selected_atoms
                )
            lambda_basis = lambda_basis.keys_to_properties(self.neighbor_species_labels)
            lambda_basis = metatensor.torch.drop_blocks(
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


def cg_combine(A, B, C):
    return torch.einsum("im, in, mnp-> ip", A, B, C)


def get_cg_coefficients(l_max):
    cg_object = ClebschGordanReal()
    for l1 in range(l_max + 1):
        for l2 in range(l_max + 1):
            for L in range(abs(l1 - l2), min(l1 + l2, l_max) + 1):
                cg_object._add(l1, l2, L)
    return cg_object


class ClebschGordanReal:
    def __init__(self):
        self._cgs = {}

    def _add(self, l1, l2, L):
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
            where_almost_zero[0], where_almost_zero[1], where_almost_zero[2]
        ):
            rcg[i0, i1, i2] = 0.0

        self._cgs[(l1, l2, L)] = torch.tensor(rcg)

    def get(self, key):
        if key in self._cgs:
            return self._cgs[key]
        else:
            self._add(key[0], key[1], key[2])
            return self._cgs[key]


def _real2complex(L):
    """
    Computes a matrix that can be used to convert from real to complex-valued
    spherical harmonics(coefficients) of order L.

    It's meant to be applied to the left, ``real2complex @ [-L..L]``.
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


def _complex_clebsch_gordan_matrix(l1, l2, L):
    if np.abs(l1 - l2) > L or np.abs(l1 + l2) < L:
        return np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * L + 1), dtype=np.double)
    else:
        return wigners.clebsch_gordan_array(l1, l2, L)


class FakeVectorBasis(torch.nn.Module):
    # fake class to make torchscript work

    def forward(
        self,
        interatomic_vectors,
        centers,
        neighbors,
        species,
        structures,
        atom_index_in_structure,
        selected_atoms: Optional[Labels],
    ) -> torch.Tensor:
        return torch.tensor(0)


class FakeSphericalExpansion(torch.nn.Module):
    # fake class to make torchscript work

    def forward(
        self,
        interatomic_vectors,
        centers,
        neighbors,
        species,
        structures,
        atom_index_in_structure,
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
