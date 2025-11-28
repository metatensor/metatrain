import torch

from metatrain.experimental.phace.utils import systems_to_batch
from .precomputations import Precomputer
from .message_passing import InvariantMessagePasser, EquivariantMessagePasser
from .tensor_product import TensorProduct
from .cg import get_cg_coefficients
from .cg_iterator import CGIterator
from .center_embedding import embed_centers, embed_centers_tensor_map
from typing import List

import torch
from torch.func import grad, functional_call


class BaseModel(torch.nn.Module):
    def __init__(self, hypers, atomic_types) -> None:
        super().__init__()
        self.atomic_types = atomic_types
        self.hypers = hypers
        self.mp_scaling = hypers["mp_scaling"]
        self.nu_max = hypers["max_correlation_order_per_layer"]
        self.nu_scaling = hypers["nu_scaling"]

        self.num_message_passing_layers = hypers["num_message_passing_layers"]
        if self.num_message_passing_layers < 1:
            raise ValueError("Number of message-passing layers must be at least 1")

        # Embedding of the atomic types
        n_channels = hypers["num_element_channels"]

        # A buffer that maps atomic types to indices in the embeddings
        species_to_species_index = torch.zeros(
            (max(self.atomic_types) + 1,), dtype=torch.int
        )
        species_to_species_index[self.atomic_types] = torch.arange(
            len(self.atomic_types), dtype=torch.int
        )
        self.register_buffer("species_to_species_index", species_to_species_index)

        self.embeddings = torch.nn.Embedding(len(self.atomic_types), n_channels)

        # A module that precomputes quantities that are useful in all message-passing
        # steps (spherical harmonics, distances)
        self.precomputer = Precomputer(
            max_eigenvalue=hypers["radial_basis"]["max_eigenvalue"],
            cutoff=hypers["cutoff"],
            cutoff_width=hypers["cutoff_width"],
            scale=hypers["radial_basis"]["scale"],
            optimizable_lengthscales=hypers["radial_basis"][
                "optimizable_lengthscales"
            ],
            all_species=self.atomic_types,
            use_sphericart=hypers["use_sphericart"]
        )

        # The message passing is invariant for the first layer
        self.invariant_message_passer = InvariantMessagePasser(
            self.atomic_types,
            self.mp_scaling,
            hypers["disable_nu_0"],
            self.precomputer.n_max_l,
            hypers["num_element_channels"],
        )

        self.atomic_types = self.atomic_types
        n_max = self.precomputer.n_max_l
        self.l_max = len(n_max) - 1
        self.k_max_l = [
            n_channels * n_max[l]
            for l in range(self.l_max + 1)  # noqa: E741
        ]
        self.k_max_l_max = [0] * (self.l_max + 1)
        previous = 0
        for l in range(self.l_max, -1, -1):
            self.k_max_l_max[l] = self.k_max_l[l] - previous
            previous = self.k_max_l[l]

        cgs = get_cg_coefficients(self.l_max)
        cgs = {
            str(l1) + "_" + str(l2) + "_" + str(L): tensor
            for (l1, l2, L), tensor in cgs._cgs.items()
        }

        tensor_product = TensorProduct(self.k_max_l)

        self.cg_iterator = CGIterator(
            tensor_product,
            self.nu_max - 1,
        )

        # Subsequent message-passing layers
        equivariant_message_passers: List[EquivariantMessagePasser] = []
        generalized_cg_iterators: List[CGIterator] = []
        for _ in range(self.num_message_passing_layers - 1):
            equivariant_message_passer = EquivariantMessagePasser(
                self.precomputer.n_max_l,
                self.hypers["num_element_channels"],
                tensor_product,
                self.mp_scaling,
            )
            equivariant_message_passers.append(equivariant_message_passer)
            generalized_cg_iterator = CGIterator(
                tensor_product,
                self.nu_max - 1,
            )
            generalized_cg_iterators.append(generalized_cg_iterator)
        self.equivariant_message_passers = torch.nn.ModuleList(
            equivariant_message_passers
        )
        self.generalized_cg_iterators = torch.nn.ModuleList(generalized_cg_iterators)

        self.energy_linear = torch.nn.Linear(self.k_max_l[0], 1)

    def forward(self, structures_as_list) -> torch.Tensor:
        structures = systems_to_batch(structures_as_list)

        n_atoms = len(structures["positions"])

        # precomputation of distances and spherical harmonics
        spherical_harmonics, radial_basis = self.precomputer(
            positions=structures["positions"],
            cells=structures["cells"],
            species=structures["species"],
            cell_shifts=structures["cell_shifts"],
            pairs=structures["pairs"],
            structure_pairs=structures["structure_pairs"],
            structure_offsets=structures["structure_offsets"],
            center_species=structures["species"][structures["pairs"][:, 0]],
            neighbor_species=structures["species"][structures["pairs"][:, 1]],
        )
        
        # scaling the spherical harmonics in this way makes sure that each successive
        # body-order is scaled by the same factor
        spherical_harmonics = [sh * self.nu_scaling for sh in spherical_harmonics]

        # calculate the center embeddings; these are shared across all layers
        center_species_indices = self.species_to_species_index[structures["species"]]
        center_embeddings = self.embeddings(center_species_indices)

        initial_features = torch.ones(
            (n_atoms, 1, self.k_max_l[0]), dtype=structures["positions"].dtype, device=structures["positions"].device
        )
        initial_element_embedding = embed_centers_tensor_map(
            [initial_features], center_embeddings
        )[0]
        # now they are all the same as the center embeddings

        # ACE-like features
        spherical_expansion = self.invariant_message_passer(
            radial_basis,
            spherical_harmonics,
            structures["structure_offsets"][structures["structure_pairs"]]
            + structures["pairs"][:, 0],
            structures["structure_offsets"][structures["structure_pairs"]]
            + structures["pairs"][:, 1],
            n_atoms,
            initial_element_embedding,
        )

        features = [spherical_expansion[l] for l in range(self.l_max + 1)]
        features = self.cg_iterator(features)

        # message passing
        for message_passer, generalized_cg_iterator in zip(
            self.equivariant_message_passers,
            self.generalized_cg_iterators,
            strict=False,
        ):
            embedded_features = embed_centers(features, center_embeddings)
            mp_features = message_passer(
                radial_basis,
                spherical_harmonics,
                structures["structure_offsets"][structures["structure_pairs"]]
                + structures["pairs"][:, 0],
                structures["structure_offsets"][structures["structure_pairs"]]
                + structures["pairs"][:, 1],
                embedded_features,
            )
            iterated_features = generalized_cg_iterator(mp_features)
            features = iterated_features

        # TODO: change position?
        features = embed_centers(features, center_embeddings)

        # final energy prediction
        energies = self.energy_linear(features[0])

        return energies


class GradientModel(torch.nn.Module):
    def __init__(self, hypers, atomic_types) -> None:
        super().__init__()
        # We wrap layers in a sub-module so we can functionalize 'self.net' 
        # inside 'forward' without triggering recursive calls to 'self.forward'.
        self.module = BaseModel(hypers, atomic_types)
        self.k_max_l = self.module.k_max_l

    def forward(self, structures_as_list):

        positions = [s[0] for s in structures_as_list]
        strains = [torch.eye(3, device=s[0].device, dtype=s[0].dtype) for s in structures_as_list]
        tensors_to_differentiate = positions + strains

        for t in tensors_to_differentiate:
            t.requires_grad = True

        positions = tensors_to_differentiate[:len(structures_as_list)]
        strains = tensors_to_differentiate[len(structures_as_list):]
        for i in range(len(structures_as_list)):
            structures_as_list[i][0] = positions[i] @ strains[i]
            structures_as_list[i][2] = structures_as_list[i][2] @ strains[i]

        # compute_batch_val_and_grad = grad(compute_energy, argnums=2, has_aux=True)
        # gradients, energies = compute_batch_val_and_grad(params, buffers, tensors_to_differentiate)

        energies = self.module(structures_as_list)
        gradients = torch.autograd.grad(
            outputs=energies.sum(),
            inputs=tensors_to_differentiate,
            create_graph=True,
        )

        forces = -torch.concatenate(gradients[:len(structures_as_list)])
        # virials = -torch.stack(gradients[len(structures_as_list):])

        return energies, forces
