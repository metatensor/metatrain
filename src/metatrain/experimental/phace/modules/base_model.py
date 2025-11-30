from typing import Dict, List

import numpy as np
import torch
from torch.func import functional_call, grad

from metatrain.experimental.phace.utils import systems_to_batch

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
        self.mp_scaling = hypers["mp_scaling"]
        self.nu_max = hypers["max_correlation_order_per_layer"]
        self.nu_scaling = hypers["nu_scaling"]
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
            self.mp_scaling,
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
                self.mp_scaling,
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

    def forward(self, structures_as_list) -> torch.Tensor:
        device = structures_as_list[0][0].device
        if self.U_dict[0].device != device:
            self.U_dict = {key: U.to(device) for key, U in self.U_dict.items()}
        dtype = structures_as_list[0][0].dtype
        if self.U_dict[0].dtype != dtype:
            self.U_dict = {key: U.to(dtype) for key, U in self.U_dict.items()}

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
            (n_atoms, 1, self.k_max_l[0]),
            dtype=structures["positions"].dtype,
            device=structures["positions"].device,
        )
        initial_element_embedding = embed_centers(
            [initial_features], center_embeddings
        )[0]
        # now they are all the same as the center embeddings

        # ACE-like features
        features = self.invariant_message_passer(
            radial_basis,
            spherical_harmonics,
            structures["structure_offsets"][structures["structure_pairs"]]
            + structures["pairs"][:, 0],
            structures["structure_offsets"][structures["structure_pairs"]]
            + structures["pairs"][:, 1],
            n_atoms,
            initial_element_embedding,
        )

        split_features = split_up_features(features, self.k_max_l)
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
                structures["structure_offsets"][structures["structure_pairs"]]
                + structures["pairs"][:, 0],
                structures["structure_offsets"][structures["structure_pairs"]]
                + structures["pairs"][:, 1],
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
        return_dict["features"] = {L: tensor for L, tensor in enumerate(features)}

        last_layer_feature_dict: Dict[str, List[torch.Tensor]] = {}
        for output_name, layer in self.heads.items():
            last_layer_features = features
            last_layer_features[0] = layer(last_layer_features[0])  # only L=0
            last_layer_feature_dict[output_name] = last_layer_features

        for output_name, layer in self.last_layers.items():
            output: Dict[int, torch.Tensor] = {}
            for L, layer_L in layer.items():
                output[int(L)] = layer_L(last_layer_feature_dict[output_name][int(L)])
            return_dict[output_name] = output

        for output_name, llf in last_layer_feature_dict.items():
            return_dict[f"{output_name}__llf"] = {L: t for L, t in enumerate(llf)}

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
                            self.k_max_l[0], len(target_info.layout.block().properties)
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
                L = int(key_values[0])  # S = int(key_values[1]) is ignored here
                irreps.append(L)
            self.last_layers[target_name] = torch.nn.ModuleDict(
                {
                    str(L): Linear(
                        self.k_max_l[0],
                        len(target_info.layout.block({"o3_lambds": L}).properties),
                    )
                    for L in irreps
                }
            )


class GradientModel(torch.nn.Module):
    def __init__(self, hypers, dataset_info) -> None:
        super().__init__()
        self.module = BaseModel(hypers, dataset_info)

    def forward(
        self,
        structures_as_list: List[List[torch.Tensor]],
        outputs_to_take_gradients_of: List[str],
    ):
        if len(outputs_to_take_gradients_of) == 0:
            return self.module(structures_as_list)

        def compute_energy(params, buffers, input_tensors, output_name):
            positions = input_tensors[: len(structures_as_list)]
            strains = input_tensors[len(structures_as_list) :]
            for i in range(len(structures_as_list)):
                structures_as_list[i][0] = positions[i] @ strains[i]
                structures_as_list[i][2] = structures_as_list[i][2] @ strains[i]
            predictions = functional_call(
                self.module, (params, buffers), (structures_as_list,)
            )
            return predictions[output_name][0].sum(), predictions

        compute_val_and_grad = grad(compute_energy, argnums=2, has_aux=True)

        params = dict(self.module.named_parameters())
        buffers = dict(self.module.named_buffers())
        positions = [s[0] for s in structures_as_list]
        strains = [
            torch.eye(3, device=s[0].device, dtype=s[0].dtype)
            for s in structures_as_list
        ]
        tensors_to_differentiate = positions + strains

        all_gradients = {}
        for output_name in outputs_to_take_gradients_of:
            gradients, predictions = compute_val_and_grad(
                params, buffers, tensors_to_differentiate, output_name
            )
            all_gradients[f"{output_name}__for"] = -torch.concatenate(
                gradients[: len(structures_as_list)]
            )
            all_gradients[f"{output_name}__vir"] = -torch.stack(
                gradients[len(structures_as_list) :]
            )

        predictions.update(all_gradients)
        return predictions
