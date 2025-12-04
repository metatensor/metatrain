from typing import Dict, List

import numpy as np
import torch
from torch.func import functional_call, grad

from .center_embedding import embed_centers
from .cg import get_cg_coefficients
from .cg_iterator import CGIterator
from .layers import Linear
from .message_passing import EquivariantMessagePasser, InvariantMessagePasser
from .precomputations import Precomputer
from .tensor_product import couple_features, split_up_features, uncouple_features


class BaseModel(torch.nn.Module):
    def __init__(self, hypers, dataset_info) -> None:
        super().__init__()
        self.atomic_types = dataset_info.atomic_types
        self.hypers = hypers
        # Store scaling values for passing to submodules
        mp_scaling_value = hypers["mp_scaling"]
        nu_scaling_value = hypers["nu_scaling"]
        # Register scaling factors as buffers for efficiency
        self.register_buffer("mp_scaling", torch.tensor(mp_scaling_value))
        self.nu_max = hypers["max_correlation_order_per_layer"]
        self.register_buffer("nu_scaling", torch.tensor(nu_scaling_value))
        self.head_num_layers = hypers["head_num_layers"]
        self.spherical_linear_layers = hypers["spherical_linear_layers"]

        # A module that precomputes quantities that are useful in all message-passing
        # steps (spherical harmonics, distances)
        self.precomputer = Precomputer(
            max_eigenvalue=hypers["radial_basis"]["max_eigenvalue"],
            cutoff=hypers["cutoff"],
            cutoff_width=hypers["cutoff_width"],
            scale=hypers["radial_basis"]["scale"],
            optimizable_lengthscales=hypers["radial_basis"]["optimizable_lengthscales"],
            all_species=self.atomic_types,
            use_sphericart=hypers["use_sphericart"],
        )

        n_max = self.precomputer.n_max_l
        self.l_max = len(n_max) - 1
        n_channels = hypers["num_element_channels"]
        if hypers["force_rectangular"]:
            self.k_max_l = [n_channels * n_max[0]] * (self.l_max + 1)
        else:
            self.k_max_l = [
                n_channels * n_max[l]
                for l in range(self.l_max + 1)  # noqa: E741
            ]

        ################
        cg_calculator = get_cg_coefficients(2 * ((self.l_max + 1) // 2))
        self.padded_l_list = [2 * ((l + 1) // 2) for l in range(self.l_max + 1)]  # noqa: E741
        U_dict = {}
        for padded_l in np.unique(self.padded_l_list):
            cg_tensors = [
                cg_calculator._cgs[(padded_l // 2, padded_l // 2, L)]
                for L in range(padded_l + 1)
            ]
            U = torch.concatenate(
                [cg_tensor for cg_tensor in cg_tensors], dim=2
            ).reshape((padded_l + 1) ** 2, (padded_l + 1) ** 2)
            assert torch.allclose(
                U @ U.T, torch.eye((padded_l + 1) ** 2, dtype=U.dtype)
            )
            assert torch.allclose(
                U.T @ U, torch.eye((padded_l + 1) ** 2, dtype=U.dtype)
            )
            U_dict[int(padded_l)] = U
        self.U_dict = U_dict
        ################

        self.num_message_passing_layers = hypers["num_message_passing_layers"]
        if self.num_message_passing_layers < 1:
            raise ValueError("Number of message-passing layers must be at least 1")

        # A buffer that maps atomic types to indices in the embeddings
        species_to_species_index = torch.zeros(
            (max(self.atomic_types) + 1,), dtype=torch.int
        )
        species_to_species_index[self.atomic_types] = torch.arange(
            len(self.atomic_types), dtype=torch.int
        )
        self.register_buffer("species_to_species_index", species_to_species_index)

        self.embeddings = torch.nn.Embedding(len(self.atomic_types), n_channels)

        # The message passing is invariant for the first layer
        self.invariant_message_passer = InvariantMessagePasser(
            self.atomic_types,
            mp_scaling_value,
            hypers["disable_nu_0"],
            self.precomputer.n_max_l,
            self.k_max_l,
        )

        cgs = get_cg_coefficients(self.l_max)
        cgs = {
            str(l1) + "_" + str(l2) + "_" + str(L): tensor
            for (l1, l2, L), tensor in cgs._cgs.items()
        }

        self.cg_iterator = CGIterator(
            self.k_max_l, self.nu_max - 1, self.spherical_linear_layers
        )

        # Subsequent message-passing layers
        equivariant_message_passers: List[EquivariantMessagePasser] = []
        generalized_cg_iterators: List[CGIterator] = []
        for _ in range(self.num_message_passing_layers - 1):
            equivariant_message_passer = EquivariantMessagePasser(
                self.precomputer.n_max_l,
                self.k_max_l,
                mp_scaling_value,
                self.spherical_linear_layers,
            )
            equivariant_message_passers.append(equivariant_message_passer)
            generalized_cg_iterator = CGIterator(
                self.k_max_l, self.nu_max - 1, self.spherical_linear_layers
            )
            generalized_cg_iterators.append(generalized_cg_iterator)
        self.equivariant_message_passers = torch.nn.ModuleList(
            equivariant_message_passers
        )
        self.generalized_cg_iterators = torch.nn.ModuleList(generalized_cg_iterators)

        self.head_types = self.hypers["heads"]
        self.heads = torch.nn.ModuleDict()
        self.last_layers = torch.nn.ModuleDict()
        for target_name, target_info in dataset_info.targets.items():
            self._add_output(target_name, target_info)

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[int, torch.Tensor]]:
        """
        Forward pass of the base model.

        :param batch: Dictionary containing batched tensors:
            - positions: stacked positions of all atoms [N_total, 3]
            - cells: stacked unit cells [N_structures, 3, 3]
            - species: atomic types of all atoms [N_total]
            - cell_shifts: cell shift vectors for all pairs [N_pairs, 3]
            - center_indices: global center indices for all pairs [N_pairs]
            - neighbor_indices: global neighbor indices for all pairs [N_pairs]
            - structure_pairs: structure index for each pair [N_pairs]
        :return: Dictionary of predictions
        """
        device = batch["positions"].device
        if self.U_dict[0].device != device:
            self.U_dict = {key: U.to(device) for key, U in self.U_dict.items()}
        dtype = batch["positions"].dtype
        if self.U_dict[0].dtype != dtype:
            self.U_dict = {key: U.to(dtype) for key, U in self.U_dict.items()}

        n_atoms = batch["positions"].size(0)

        # precomputation of distances and spherical harmonics
        spherical_harmonics, radial_basis = self.precomputer(
            positions=batch["positions"],
            cells=batch["cells"],
            species=batch["species"],
            cell_shifts=batch["cell_shifts"],
            center_indices=batch["center_indices"],
            neighbor_indices=batch["neighbor_indices"],
            structure_pairs=batch["structure_pairs"],
            center_species=batch["species"][batch["center_indices"]],
            neighbor_species=batch["species"][batch["neighbor_indices"]],
        )

        # scaling the spherical harmonics in this way makes sure that each successive
        # body-order is scaled by the same factor
        spherical_harmonics = [sh * self.nu_scaling for sh in spherical_harmonics]

        # calculate the center embeddings; these are shared across all layers
        center_species_indices = self.species_to_species_index[batch["species"]]
        center_embeddings = self.embeddings(center_species_indices)

        initial_features = torch.ones(
            (n_atoms, 1, self.k_max_l[0]),
            dtype=batch["positions"].dtype,
            device=batch["positions"].device,
        )
        initial_element_embedding = embed_centers(
            [initial_features], center_embeddings
        )[0]
        # now they are all the same as the center embeddings

        # ACE-like features
        features = self.invariant_message_passer(
            radial_basis,
            spherical_harmonics,
            batch["center_indices"],
            batch["neighbor_indices"],
            n_atoms,
            initial_element_embedding,
        )

        split_features = split_up_features(features, self.k_max_l)
        features = []
        for l in range(self.l_max + 1):  # noqa: E741
            features.append(
                uncouple_features(
                    split_features[l],
                    self.U_dict[self.padded_l_list[l]],
                    self.padded_l_list[l],
                )
            )

        features = self.cg_iterator(features, self.U_dict)

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
                batch["center_indices"],
                batch["neighbor_indices"],
                embedded_features,
                self.U_dict,
            )
            iterated_features = generalized_cg_iterator(mp_features, self.U_dict)
            features = iterated_features

        coupled_features: List[List[torch.Tensor]] = []
        for l in range(self.l_max + 1):  # noqa: E741
            coupled_features.append(
                couple_features(
                    features[l],
                    self.U_dict[self.padded_l_list[l]],
                    self.padded_l_list[l],
                )
            )
        features = []
        for l in range(self.l_max + 1):  # noqa: E741
            features.append(
                torch.concatenate(
                    [coupled_features[lp][l] for lp in range(l, self.l_max + 1)], dim=-1
                )
            )

        # TODO: change position?
        features = embed_centers(features, center_embeddings)

        # final predictions
        return_dict: Dict[str, Dict[int, torch.Tensor]] = {}
        return_dict["features"] = {l: tensor for l, tensor in enumerate(features)}  # noqa: E741

        last_layer_feature_dict: Dict[str, List[torch.Tensor]] = {}
        for output_name, layer in self.heads.items():
            last_layer_features = features
            last_layer_features[0] = layer(last_layer_features[0])  # only L=0
            last_layer_feature_dict[output_name] = last_layer_features

        for output_name, layer in self.last_layers.items():
            output: Dict[int, torch.Tensor] = {}
            for l_str, layer_L in layer.items():
                l = int(l_str)  # noqa: E741
                output[l] = layer_L(last_layer_feature_dict[output_name][l])
            return_dict[output_name] = output

        for output_name, llf in last_layer_feature_dict.items():
            return_dict[f"{output_name}__llf"] = {l: t for l, t in enumerate(llf)}  # noqa: E741

        return return_dict

    def _add_output(self, target_name, target_info):
        if target_name not in self.head_types:
            if target_info.is_scalar:
                use_mlp = True  # default to MLP for scalars
            else:
                use_mlp = False  # can't use MLP for equivariants
                # TODO: the equivariant, or part of it, could be a scalar...
        else:
            # specified by the user
            use_mlp = self.head_types[target_name] == "mlp"

        if use_mlp:
            if target_info.is_spherical or target_info.is_cartesian:
                raise ValueError("MLP heads are only supported for scalar targets.")

            layers = (
                [Linear(self.k_max_l[0], self.k_max_l[0]), torch.nn.SiLU()]
                if self.head_num_layers == 1
                else [Linear(self.k_max_l[0], 4 * self.k_max_l[0]), torch.nn.SiLU()]
                + [Linear(4 * self.k_max_l[0], 4 * self.k_max_l[0]), torch.nn.SiLU()]
                * (self.head_num_layers - 2)
                + [Linear(4 * self.k_max_l[0], self.k_max_l[0]), torch.nn.SiLU()]
            )
            self.heads[target_name] = torch.nn.Sequential(*layers)
        else:
            self.heads[target_name] = torch.nn.Identity()

        if target_info.is_scalar:
            self.last_layers[target_name] = torch.nn.ModuleDict(
                {
                    "0": Linear(
                        self.k_max_l[0], len(target_info.layout.block().properties)
                    )
                }
            )
        elif target_info.is_cartesian:
            # here, we handle Cartesian targets
            # we just treat them as a spherical L=1 targets, the conversion will be
            # performed in the metatensor wrapper
            if len(target_info.layout.block().components) == 1:
                self.last_layers[target_name] = torch.nn.ModuleDict(
                    {
                        "1": Linear(
                            self.k_max_l[1], len(target_info.layout.block().properties)
                        )
                    }
                )
            else:
                raise NotImplementedError(
                    "PhACE only supports Cartesian targets with rank=1."
                )
        else:  # spherical equivariant
            irreps = []
            for key in target_info.layout.keys:
                key_values = key.values
                l = int(key_values[0])  # noqa: E741
                # s = int(key_values[1]) is ignored here
                irreps.append(l)
            self.last_layers[target_name] = torch.nn.ModuleDict(
                {
                    str(l): Linear(
                        self.k_max_l[l],
                        len(target_info.layout.block({"o3_lambda": l}).properties),
                    )
                    for l in irreps  # noqa: E741
                }
            )


class GradientModel(torch.nn.Module):
    """
    Wrapper around BaseModel that computes gradients with respect to positions and
    strain.

    Uses batched tensor representation for torch.compile compatibility.
    """

    def __init__(self, module) -> None:
        super().__init__()
        self.module = module

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        outputs_to_take_gradients_of: List[str],
    ):
        if len(outputs_to_take_gradients_of) == 0:
            return self.module(batch)

        n_structures = batch["n_atoms"].size(0)
        device = batch["positions"].device
        dtype = batch["positions"].dtype

        def compute_energy(params, buffers, positions, strains, output_name):
            # Apply strain to positions and cells
            # For each atom, get the strain matrix for its structure by indexing with
            # structure_centers
            # strains: [n_structures, 3, 3]
            # structure_centers: [n_atoms] - maps each atom to its structure index
            # positions: [n_atoms, 3]

            # Get the strain matrix for each atom: [n_atoms, 3, 3]
            atom_strains = strains[batch["structure_centers"]]

            # Apply strain to positions: pos @ strain for each atom (using einsum)
            strained_positions = torch.einsum("ij,ijk->ik", positions, atom_strains)

            # Apply strain to cells: [n_structures, 3, 3] @ [n_structures, 3, 3]
            strained_cells = torch.bmm(batch["cells"], strains)

            # Create a modified batch with strained positions and cells
            strained_batch = {
                "positions": strained_positions,
                "cells": strained_cells,
                "species": batch["species"],
                "cell_shifts": batch["cell_shifts"],
                "center_indices": batch["center_indices"],
                "neighbor_indices": batch["neighbor_indices"],
                "structure_pairs": batch["structure_pairs"],
            }

            predictions = functional_call(
                self.module, (params, buffers), (strained_batch,)
            )
            return predictions[output_name][0].sum(), predictions

        compute_val_and_grad = grad(compute_energy, argnums=(2, 3), has_aux=True)

        params = dict(self.module.named_parameters())
        buffers = dict(self.module.named_buffers())

        # Create strain tensors (one 3x3 identity per structure)
        strains = (
            torch.eye(3, device=device, dtype=dtype)
            .unsqueeze(0)
            .expand(n_structures, -1, -1)
            .clone()
        )  # [n_structures, 3, 3]

        all_gradients = {}
        for output_name in outputs_to_take_gradients_of:
            (pos_grad, strain_grads), predictions = compute_val_and_grad(
                params, buffers, batch["positions"], strains, output_name
            )
            all_gradients[f"{output_name}__for"] = {
                -1: -pos_grad  # Forces are negative gradient of energy
            }
            all_gradients[f"{output_name}__vir"] = {
                -1: -strain_grads  # Virial/stress from strain gradient
            }

        predictions.update(all_gradients)
        return predictions


class FakeGradientModel(torch.nn.Module):
    """
    Wrapper around BaseModel that does not compute gradients.

    Used during inference when gradients are not needed.
    """

    def __init__(self, module) -> None:
        super().__init__()
        self.module = module

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        outputs_to_take_gradients_of: List[str],
    ):
        return self.module(batch)
