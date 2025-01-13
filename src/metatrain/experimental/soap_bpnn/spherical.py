"""Modules to allow SOAP-BPNN to fit arbitrary spherical tensor targets."""

import copy
from typing import List

import rascaline.torch
import torch
from metatensor.torch import Labels
from metatensor.torch.atomistic import System


class ScalarBasis(torch.nn.Module):
    """
    A dummy module to trick torchscript (see model.py).
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, systems: List[System]) -> torch.Tensor:
        return torch.Tensor()


class VectorBasis(torch.nn.Module):
    """
    This module creates a basis of 3 vectors for each atomic environment.

    In practice, this is done by contracting a l=1 spherical expansion.
    """

    def __init__(self, atomic_types, soap_hypers) -> None:
        super().__init__()
        self.atomic_types = atomic_types
        soap_vector_hypers = copy.deepcopy(soap_hypers)
        soap_vector_hypers["max_angular"] = 1
        self.soap_calculator = rascaline.torch.SphericalExpansion(
            radial_basis={"Gto": {}}, **soap_vector_hypers
        )
        self.neighbor_species_labels = Labels(
            names=["neighbor_type"],
            values=torch.tensor(self.atomic_types).reshape(-1, 1),
        )
        self.contractor = torch.nn.Parameter(
            torch.randn((soap_vector_hypers["max_radial"] * len(self.atomic_types), 3))
        )
        # this optimizable basis seems to work much better than the fixed one here:
        # self.register_buffer(
        #     "contractor",
        #     torch.randn((soap_vector_hypers["max_radial"]*len(self.atomic_types), 3)),
        # )

    def forward(self, systems: List[System]) -> torch.Tensor:
        device = systems[0].positions.device
        if self.neighbor_species_labels.device != device:
            self.neighbor_species_labels = self.neighbor_species_labels.to(device)

        spherical_expansion = self.soap_calculator(systems)

        # by calling these two in the same order that they are called in the main
        # model, we should ensure that the order of the samples is the same
        spherical_expansion = spherical_expansion.keys_to_properties(
            self.neighbor_species_labels
        )
        spherical_expansion = spherical_expansion.keys_to_samples("center_type")

        basis_vectors = (
            spherical_expansion.block({"o3_lambda": 1}).values @ self.contractor
        )

        return basis_vectors  # [n_atoms, 3(yzx), 3]


class TensorBasis(torch.nn.Module):
    """
    Creates a basis of spherical tensors for each atomic environment, starting from
    a basis of 3 vectors.
    """

    pass
