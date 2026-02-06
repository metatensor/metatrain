"""Modules to allow SOAP-BPNN to fit arbitrary spherical tensor targets."""

import copy
from typing import Any, Dict, List, Optional, Tuple

import metatensor.torch as mts
import numpy as np
import sphericart.torch
import torch
import wigners
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.learn.nn import Linear as LinearMap
from spex import SphericalExpansion

from ..documentation import SOAPConfig


def _build_spex_hypers(
    soap_hypers: SOAPConfig,
    max_angular: int,
    atomic_types: List[int],
    legacy: bool,
) -> Dict[str, Any]:
    """
    Build the hypers dictionary for SphericalExpansion.

    :param soap_hypers: dictionary with the SOAP hyper-parameters.
    :param max_angular: maximum angular momentum to be used.
    :param atomic_types: list of atomic types in the dataset.
    :param legacy: if True, uses the legacy implementation without chemical embedding.
    :return: dictionary with the hyper-parameters for SphericalExpansion.
    """
    soap_hypers = copy.deepcopy(soap_hypers)

    species_def: Dict[str, Any]  # make mypy happy
    if legacy:
        species_def = {"Orthogonal": {"species": atomic_types}}
    else:
        # hardcoded to 4 (the literature would suggest 4 is enough)
        species_def = {
            "Alchemical": {
                "pseudo_species": 4,
                "total_species": len(atomic_types),
            }
        }

    spex_soap_hypers = {
        "cutoff": soap_hypers["cutoff"]["radius"],
        "max_angular": max_angular,
        "radial": {
            "LaplacianEigenstates": {
                "max_radial": soap_hypers["max_radial"],
            }
        },
        "angular": "SphericalHarmonics",
        "cutoff_function": {"ShiftedCosine": {"width": soap_hypers["cutoff"]["width"]}},
        "species": species_def,
    }
    return spex_soap_hypers


def _sort_tensor_blocks_like_atoms(
    tensor_map: TensorMap,
    structures: torch.Tensor,
) -> torch.Tensor:
    """
    Given a TensorMap where each block has samples labeled by (system, atom),
    return a single values tensor sorted to follow the ordering implied by
    `structures` and an in-structure atom index.

    This is used in the legacy branches instead of
    `keys_to_samples("center_type")` for performance reasons.

    :param tensor_map: TensorMap with blocks labeled by (system, atom).
    :param structures: (num_atoms,) tensor with the index of the structure each
        atom belongs to.
    :return: a tensor of shape (num_atoms, ...) with the values sorted to follow
        the ordering implied by `structures` and an in-structure atom index.
    """
    device = structures.device

    # Concatenate values and sample indices from all blocks
    all_values = torch.concatenate([b.values for b in tensor_map.blocks()])
    all_system_indices = torch.concatenate(
        [b.samples.values[:, 0] for b in tensor_map.blocks()]
    )
    all_atom_indices = torch.concatenate(
        [b.samples.values[:, 1] for b in tensor_map.blocks()]
    )

    # system_sizes and offsets assume that systems are labeled from 0..N-1,
    # which is consistent with how `structures` is constructed in this module.
    system_sizes = torch.bincount(structures, minlength=len(torch.unique(structures)))
    system_offsets = torch.cat(
        [
            torch.tensor([0], device=device, dtype=structures.dtype),
            torch.cumsum(system_sizes, dim=0)[:-1],
        ]
    )

    overall_atom_indices = system_offsets[all_system_indices] + all_atom_indices
    sorting_indices = torch.argsort(overall_atom_indices)

    return all_values[sorting_indices]


def _build_spherical_basis_tensormap(
    expansion: torch.Tensor,
    o3_lambda: int,
    species: torch.Tensor,
    structures: torch.Tensor,
    atom_index_in_structure: torch.Tensor,
    legacy: bool,
) -> TensorMap:
    """
    Build a TensorMap from a spherical expansion tensor.

    Handles both legacy (one block per center species) and modern (single block)
    layouts.

    :param expansion: tensor of shape ``(n_atoms, 2*o3_lambda+1, n_features)``.
        For the modern path the caller must already have applied the center
        encoding (multiplication by the embedding).
    :param o3_lambda: angular momentum order.
    :param species: ``(n_atoms,)`` atomic species tensor.
    :param structures: ``(n_atoms,)`` structure index per atom.
    :param atom_index_in_structure: ``(n_atoms,)`` atom index within structure.
    :param legacy: whether to use the legacy multi-block layout.
    :return: TensorMap ready for contraction.
    """
    device = expansion.device

    o3_mu_labels = Labels(
        names=["o3_mu"],
        values=torch.arange(
            -o3_lambda, o3_lambda + 1, dtype=torch.long, device=device
        ).unsqueeze(1),
    )
    property_labels = Labels(
        names=["property"],
        values=torch.arange(expansion.shape[2], device=device).unsqueeze(1),
    )

    if legacy:
        unique_center_species = torch.unique(species)
        blocks: List[TensorBlock] = []

        for s in unique_center_species:
            mask = species == s
            blocks.append(
                TensorBlock(
                    values=expansion[mask],
                    samples=Labels(
                        names=["system", "atom"],
                        values=torch.stack(
                            [structures[mask], atom_index_in_structure[mask]], dim=1
                        ),
                    ),
                    components=[o3_mu_labels],
                    properties=property_labels,
                )
            )

        return TensorMap(
            keys=Labels(
                names=["o3_lambda", "o3_sigma", "center_type"],
                values=torch.tensor(
                    [[o3_lambda, 1, int(s)] for s in unique_center_species],
                    device=device,
                ),
            ),
            blocks=blocks,
        )
    else:
        return TensorMap(
            keys=Labels(
                ["o3_lambda", "o3_sigma"],
                torch.tensor([[o3_lambda, 1]], device=device),
            ),
            blocks=[
                TensorBlock(
                    values=expansion,
                    samples=Labels(
                        names=["system", "atom"],
                        values=torch.stack(
                            [structures, atom_index_in_structure], dim=1
                        ),
                    ),
                    components=[o3_mu_labels],
                    properties=property_labels,
                )
            ],
        )


class VectorBasis(torch.nn.Module):
    """
    This module creates a basis of 3 vectors for each atomic environment.

    In practice, this is done by contracting an l=1 spherical expansion.

    :param atomic_types: list of atomic types in the dataset.
    :param soap_hypers: dictionary with the SOAP hyper-parameters.
    :param legacy: if True, uses the legacy implementation without chemical embedding.
    """

    def __init__(
        self,
        atomic_types: List[int],
        soap_hypers: SOAPConfig,
        legacy: bool,
    ) -> None:
        super().__init__()

        self.legacy = legacy
        self.atomic_types = atomic_types

        spex_soap_hypers = _build_spex_hypers(
            soap_hypers=soap_hypers,
            max_angular=1,
            atomic_types=self.atomic_types,
            legacy=self.legacy,
        )
        self.soap_calculator = SphericalExpansion(**spex_soap_hypers)

        self.neighbor_species_labels = Labels(
            names=["neighbor_type"],
            values=torch.tensor(self.atomic_types).reshape(-1, 1),
        )

        l1_n_radial = self.soap_calculator.radial.n_per_l[1]

        if self.legacy:
            # Legacy mode: no chemical embedding, contraction done via LinearMap
            self.center_encoding = torch.nn.Identity()
            self.contraction_for_tensors = torch.nn.Identity()
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
                in_features=l1_n_radial * len(self.atomic_types),
                out_features=3,
                bias=False,
                out_properties=[Labels.range("basis", 3) for _ in self.atomic_types],
            )
        else:
            # modern version: chemical embedding, contraction as a pure tensor op
            embedding_dim = l1_n_radial * 4
            self.center_encoding = torch.nn.Embedding(
                num_embeddings=len(self.atomic_types),
                embedding_dim=embedding_dim,
            )
            # here, an optimizable basis seems to work much better than a fixed one
            self.contraction_for_tensors = torch.nn.Linear(
                in_features=embedding_dim,
                out_features=3,
                bias=False,
            )
            # kept for TorchScript parity with the legacy path
            self.contraction = FakeLinearMap()

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

        if self.neighbor_species_labels.values.device != device:
            self.neighbor_species_labels = self.neighbor_species_labels.to(device)

        # only l=1 tensor
        l1_spherical_expansion = self.soap_calculator(
            interatomic_vectors,
            centers,
            neighbors,
            species,
        )[1]

        # [center, o3_mu, features]
        l1_spherical_expansion = l1_spherical_expansion.reshape(
            l1_spherical_expansion.shape[0],
            l1_spherical_expansion.shape[1],
            l1_spherical_expansion.shape[2] * l1_spherical_expansion.shape[3],
        )

        if not self.legacy:
            l1_spherical_expansion = l1_spherical_expansion * (
                self.center_encoding(species).unsqueeze(1)
            )

        l1_tensor_map = _build_spherical_basis_tensormap(
            expansion=l1_spherical_expansion,
            o3_lambda=1,
            species=species,
            structures=structures,
            atom_index_in_structure=atom_index_in_structure,
            legacy=self.legacy,
        )

        if selected_atoms is not None:
            l1_tensor_map = mts.slice(l1_tensor_map, "samples", selected_atoms)

        if self.legacy:
            contracted = self.contraction(l1_tensor_map)
            return _sort_tensor_blocks_like_atoms(contracted, structures=structures)
        else:
            return self.contraction_for_tensors(l1_tensor_map.block().values)


class LambdaBasis(torch.nn.Module):
    """
    Contracted spherical expansion basis at a given angular order.

    Computes a spherical expansion at angular order ``o3_lambda``, wraps it into a
    TensorMap, and contracts it down to ``2*o3_lambda+1`` features per component,
    producing a tensor of shape ``(n_atoms, 2*o3_lambda+1, 2*o3_lambda+1)``.

    :param atomic_types: list of atomic types in the dataset.
    :param soap_hypers: SOAP hyperparameters.
    :param o3_lambda: angular momentum order (must be > 1).
    :param legacy: whether to use the legacy multi-block layout.
    """

    def __init__(
        self,
        atomic_types: List[int],
        soap_hypers: SOAPConfig,
        o3_lambda: int,
        legacy: bool,
    ) -> None:
        super().__init__()

        self.legacy = legacy
        self.o3_lambda = o3_lambda

        spex_soap_hypers = _build_spex_hypers(
            soap_hypers=soap_hypers,
            max_angular=o3_lambda,
            atomic_types=atomic_types,
            legacy=legacy,
        )
        self.spex_calculator = SphericalExpansion(**spex_soap_hypers)

        if legacy:
            n_radial = self.spex_calculator.radial.n_per_l[o3_lambda]
            self.spex_contraction = LinearMap(
                in_keys=Labels(
                    names=["o3_lambda", "o3_sigma", "center_type"],
                    values=torch.stack(
                        [
                            torch.tensor([o3_lambda] * len(atomic_types)),
                            torch.tensor([1] * len(atomic_types)),
                            torch.tensor(atomic_types),
                        ],
                        dim=1,
                    ),
                ),
                in_features=n_radial * len(atomic_types),
                out_features=2 * o3_lambda + 1,
                bias=False,
                out_properties=[
                    Labels.range("lambda_basis", 2 * o3_lambda + 1)
                    for _ in atomic_types
                ],
            )
            self.spex_contraction_for_tensors = torch.nn.Identity()
            self.center_encoding = torch.nn.Identity()
        else:
            n_radial = self.spex_calculator.radial.n_per_l[o3_lambda]
            self.spex_contraction_for_tensors = torch.nn.Linear(
                in_features=n_radial * 4,
                out_features=2 * o3_lambda + 1,
                bias=False,
            )
            self.spex_contraction = FakeLinearMap()
            self.center_encoding = torch.nn.Embedding(
                num_embeddings=len(atomic_types),
                embedding_dim=n_radial * 4,
            )

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
        Compute the contracted lambda basis.

        :param interatomic_vectors: (num_edges, 3) tensor with the vectors from
            each center to each neighbor.
        :param centers: (num_edges,) tensor with the indices of the center atoms.
        :param neighbors: (num_edges,) tensor with the indices of the neighbor
            atoms.
        :param species: (num_atoms,) tensor with the atomic species of each atom.
        :param structures: (num_atoms,) tensor with the index of the structure each
            atom belongs to.
        :param atom_index_in_structure: (num_atoms,) tensor with the index of each atom
            within its structure.
        :param selected_atoms: optional Labels object to select a subset of atoms.
        :return: tensor of shape ``(n_atoms, 2*o3_lambda+1, 2*o3_lambda+1)``.
        """
        lambda_basis = self.spex_calculator(
            interatomic_vectors,
            centers,
            neighbors,
            species,
        )[self.o3_lambda]

        # [center, o3_mu, features]
        lambda_basis = lambda_basis.reshape(
            lambda_basis.shape[0],
            lambda_basis.shape[1],
            lambda_basis.shape[2] * lambda_basis.shape[3],
        )

        if not self.legacy:
            lambda_basis = lambda_basis * (self.center_encoding(species).unsqueeze(1))

        lambda_tensor_map = _build_spherical_basis_tensormap(
            expansion=lambda_basis,
            o3_lambda=self.o3_lambda,
            species=species,
            structures=structures,
            atom_index_in_structure=atom_index_in_structure,
            legacy=self.legacy,
        )

        if selected_atoms is not None:
            lambda_tensor_map = mts.slice(lambda_tensor_map, "samples", selected_atoms)

        if self.legacy:
            contracted = self.spex_contraction(lambda_tensor_map)
            return _sort_tensor_blocks_like_atoms(contracted, structures=structures)
        else:
            return self.spex_contraction_for_tensors(lambda_tensor_map.block().values)


class FakeLambdaBasis(torch.nn.Module):
    """Dummy module for TorchScript compatibility when lambda basis is disabled."""

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
    :param legacy: if True, uses the legacy implementation without chemical embedding.
    """

    cgs: Dict[str, torch.Tensor]  # torchscript needs this
    """dictionary with the Clebsch-Gordan coefficients"""

    def __init__(
        self,
        atomic_types: List[int],
        soap_hypers: SOAPConfig,
        o3_lambda: int,
        o3_sigma: int,
        add_lambda_basis: bool,
        legacy: bool,
    ) -> None:
        super().__init__()

        self.legacy = legacy
        self.o3_lambda = o3_lambda
        self.o3_sigma = o3_sigma
        self.atomic_types = atomic_types

        # Vector bases
        if self.o3_lambda > 0:
            self.vector_basis = VectorBasis(atomic_types, soap_hypers, legacy)
        else:
            self.vector_basis = FakeVectorBasis()  # needed to make torchscript work

        if self.o3_sigma == -1:
            self.vector_basis_pseudotensor = VectorBasis(
                atomic_types, soap_hypers, legacy
            )
        else:
            self.vector_basis_pseudotensor = FakeVectorBasis()  # make torchscript work

        # Spherical harmonics calculator
        if self.o3_lambda > 1:
            self.spherical_harmonics_calculator = sphericart.torch.SolidHarmonics(
                l_max=self.o3_lambda
            )
        else:
            # needed to make torchscript work
            self.spherical_harmonics_calculator = torch.nn.Identity()

        # Clebsch–Gordan coefficients
        if self.o3_lambda > 1 or self.o3_sigma == -1:
            cg_object = get_cg_coefficients(
                max(self.o3_lambda, 1)  # need at least 1 for pseudoscalar case
            )
            self.cgs = {
                f"{l1}_{l2}_{L}": cg_tensor
                for (l1, l2, L), cg_tensor in cg_object._cgs.items()
            }
        else:
            # needed to make torchscript work
            self.cgs = {}  # type: ignore[assignment]

        self.neighbor_species_labels = Labels(
            names=["neighbor_type"],
            values=torch.tensor(self.atomic_types).reshape(-1, 1),
        )

        # Optional lambda basis
        self.add_lambda_basis = add_lambda_basis and o3_lambda > 1
        if self.add_lambda_basis:
            self.lambda_basis_module = LambdaBasis(
                atomic_types, soap_hypers, o3_lambda, legacy
            )
        else:
            self.lambda_basis_module = FakeLambdaBasis()

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
        device = interatomic_vectors.device
        dtype = interatomic_vectors.dtype

        # transfer cg dict to device and dtype if needed
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
            L = self.o3_lambda
            basis = torch.empty(
                (num_atoms, 2 * L + 1, 2 * L + 1),
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

            for lam in range(L + 1):
                basis[:, :, lam] = cg_combine(
                    sh_1[:, lam * lam : (lam + 1) * (lam + 1)],
                    sh_2[
                        :,
                        (L - lam) * (L - lam) : (L - lam + 1) * (L - lam + 1),
                    ],
                    self.cgs[f"{lam}_{L - lam}_{L}"],
                )

            for lam in range(L):
                basis[:, :, L + 1 + lam] = cg_combine(
                    cg_combine(
                        sh_1[:, lam * lam : (lam + 1) * (lam + 1)],
                        sh_2[
                            :,
                            (L - lam - 1) * (L - lam - 1) : (L - lam) * (L - lam),
                        ],
                        self.cgs[f"{lam}_{L - lam - 1}_{L - 1}"],
                    ),
                    vector_3_spherical,
                    self.cgs[f"{L - 1}_1_{L}"],
                )

        if self.add_lambda_basis:
            lambda_basis_tensor = self.lambda_basis_module(
                interatomic_vectors,
                centers,
                neighbors,
                species,
                structures,
                atom_index_in_structure,
                selected_atoms,
            )
            basis = torch.cat((basis, lambda_basis_tensor), dim=-1)

        # ---- pseudotensor factor (pseudoscalar) ----
        if self.o3_sigma == -1:
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
                cg_combine(
                    vector_1_spherical,
                    vector_2_spherical,
                    self.cgs["1_1_1"],
                ),
                vector_3_spherical,
                self.cgs["1_1_0"],
            )
            basis = basis * pseudoscalar.unsqueeze(1)

        return basis  # [n_atoms, 2*o3_lambda+1, 2*o3_lambda+1]


# ----------------------------------------------------------------------------- #
# Clebsch–Gordan machinery
# ----------------------------------------------------------------------------- #


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
        r2c: Dict[int, np.ndarray] = {}
        c2r: Dict[int, np.ndarray] = {}
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
    return wigners.clebsch_gordan_array(l1, l2, L)


# ----------------------------------------------------------------------------- #
# TorchScript helper classes (dummy / placeholder modules)
# ----------------------------------------------------------------------------- #


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
    ) -> torch.Tensor:
        return torch.tensor(0)


class FakeLinearMap(torch.nn.Module):
    # fake class to make torchscript work
    def forward(
        self,
        tensor_map: TensorMap,
    ) -> TensorMap:
        return tensor_map
