"""PET trunk as ``sr``, with a spherical sidecar for paper LR / BEC.

``PETBackend`` supplies invariant node features. ``LoremBackbone`` still
builds ``spherical_features`` so ``LoremLongRangeFeaturizer`` and
``BornEffectiveChargeHead`` keep the equivariant charge / APT path. PET's
own ``long_range`` stays off — LOREM owns ``lr``.
"""

from copy import deepcopy
from typing import List, Tuple

import torch
from metatomic.torch import NeighborListOptions, System

from metatrain.pet.modules.backend import PETBackend
from metatrain.pet.modules.structures import (
    concatenate_structures as concatenate_pet_structures,
)
from metatrain.utils.architectures import get_default_hypers

from .backbone import LoremBackbone


class PetTrunk(torch.nn.Module):
    """PET node features + LOREM spherical density sidecar."""

    def __init__(
        self,
        hypers: dict,
        atomic_types: List[int],
        neighbor_list_options: NeighborListOptions,
    ) -> None:
        super().__init__()
        self.neighbor_list_options = neighbor_list_options
        self.num_features = int(hypers["num_features"])
        self.sidecar = LoremBackbone(hypers, atomic_types, neighbor_list_options)

        pet_hypers = deepcopy(get_default_hypers("pet")["model"])
        user_pet = hypers["pet"] if "pet" in hypers else {}
        pet_hypers.update(user_pet)
        pet_hypers["cutoff"] = float(hypers["cutoff"])
        pet_hypers["cutoff_width"] = float(hypers["cutoff_width"])
        pet_hypers["d_node"] = self.num_features
        # PET's own long-range branch always stays off: LOREM.lr handles the
        # long-range Coulomb message unconditionally, on top of this trunk.
        long_range = dict(pet_hypers["long_range"])
        long_range["enable"] = False
        pet_hypers["long_range"] = long_range
        self.cutoff_width_adaptive = float(pet_hypers["cutoff_width_adaptive"])
        self.backend = PETBackend(pet_hypers, atomic_types)

    def forward(
        self, systems: List[System]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return PET node features, sidecar distances, and spherical features."""
        _, distances, spherical_features = self.sidecar(systems)
        (
            positions,
            centers,
            neighbors,
            species,
            cells,
            cell_shifts,
            system_indices,
            _,
        ) = concatenate_pet_structures(systems, self.neighbor_list_options)
        batch_data = self.backend.preprocess(
            positions,
            centers,
            neighbors,
            species,
            cells,
            cell_shifts,
            system_indices,
            self.cutoff_width_adaptive,
        )
        node_features_list, _ = self.backend.calculate_features(batch_data, False)
        features = node_features_list[-1]
        return features, distances, spherical_features
