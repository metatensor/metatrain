from typing import Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import ModelOutput, System
from omegaconf import OmegaConf

from ... import ARCHITECTURE_CONFIG_PATH
from .pet.pet_torch.encoder import Encoder
from .pet.pet_torch.nef import edge_array_to_nef, get_nef_indices
from .pet.pet_torch.radial_mask import get_radial_mask
from .pet.pet_torch.structures import concatenate_structures
from .pet.pet_torch.transformer import Transformer


ARCHITECTURE_NAME = "experimental.pet_jax"
DEFAULT_HYPERS = OmegaConf.to_container(
    OmegaConf.load(ARCHITECTURE_CONFIG_PATH / f"{ARCHITECTURE_NAME}.yaml")
)
DEFAULT_MODEL_HYPERS = DEFAULT_HYPERS["model"]


class Model(torch.nn.Module):

    def __init__(self, all_species, hypers, composition_weights):
        super().__init__()

        n_species = len(all_species)

        # Handle species
        self.all_species = all_species
        self.species_to_species_index = torch.full(
            (torch.max(all_species) + 1,),
            -1,
        )
        for i, species in enumerate(all_species):
            self.species_to_species_index[species] = i
        print("Species indices:", self.species_to_species_index)
        print("Number of species:", n_species)

        self.encoder = Encoder(n_species, hypers["d_pet"])

        self.transformer = Transformer(
            hypers["d_pet"],
            4 * hypers["d_pet"],
            hypers["num_heads"],
            hypers["num_attention_layers"],
            hypers["mlp_dropout_rate"],
            hypers["attention_dropout_rate"],
        )
        self.readout = torch.nn.Linear(hypers["d_pet"], 1, bias=False)

        self.register_buffer("composition_weights", composition_weights)

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:

        # TODO: checks

        n_structures = len(systems)
        positions, centers, neighbors, species, segment_indices, edge_vectors = (
            concatenate_structures(systems)
        )
        max_edges_per_node = int(torch.max(torch.bincount(centers)))

        # Convert to NEF:
        nef_indices = get_nef_indices(centers, len(positions), max_edges_per_node)

        # Get radial mask
        r = torch.sqrt(torch.sum(edge_vectors**2, axis=-1))
        radial_mask = get_radial_mask(r, 5.0, 3.0)

        # Element indices
        element_indices_nodes = self.species_to_species_index[species]
        element_indices_centers = element_indices_nodes[centers]
        element_indices_neighbors = element_indices_nodes[neighbors]

        # Send everything to NEF:
        edge_vectors = edge_array_to_nef(edge_vectors, nef_indices, 0.0)
        radial_mask = edge_array_to_nef(radial_mask, nef_indices, 0.0)
        element_indices_centers = edge_array_to_nef(
            element_indices_centers, nef_indices, self.all_species[0]
        )
        element_indices_neighbors = edge_array_to_nef(
            element_indices_neighbors, nef_indices, self.all_species[0]
        )

        features = {
            "cartesian": edge_vectors,
            "center": element_indices_centers,
            "neighbor": element_indices_neighbors,
        }

        # Encode
        features = self.encoder(features)

        # Transformer
        features = self.transformer(features, radial_mask)

        # Readout
        edge_energies = self.readout(features)
        edge_energies = edge_energies * radial_mask[:, :, None]

        # Sum over edges
        atomic_energies = torch.sum(
            edge_energies, axis=(1, 2)
        )  # also eliminate singleton dimension 2

        # Sum over centers
        structure_energies = torch.zeros(
            n_structures, dtype=atomic_energies.dtype, device=atomic_energies.device
        )
        structure_energies.index_add_(0, segment_indices, atomic_energies)

        # TODO: use utils? use composition calculator?
        composition = torch.zeros(
            (n_structures, len(self.all_species)), device=atomic_energies.device
        )
        for number in self.all_species:
            where_number = (species == number).to(composition.dtype)
            composition[:, self.species_to_species_index[number]].index_add_(
                0, segment_indices, where_number
            )

        structure_energies = structure_energies + composition @ self.composition_weights

        return {
            next(iter(outputs.keys())): TensorMap(
                keys=Labels(
                    names=["o3_lambda", "o3_sigma"],
                    values=torch.tensor([[0, 1]], device=structure_energies.device),
                ),
                blocks=[
                    TensorBlock(
                        values=structure_energies.unsqueeze(1),
                        samples=Labels(
                            names=["structure"],
                            values=torch.arange(
                                n_structures, device=structure_energies.device
                            ).unsqueeze(1),
                        ),
                        components=[],
                        properties=Labels(
                            names=["_"],
                            values=torch.tensor(
                                [[0]], device=structure_energies.device
                            ),
                        ),
                    )
                ],
            )
        }
