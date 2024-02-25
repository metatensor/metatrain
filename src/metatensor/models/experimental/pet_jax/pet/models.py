from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp

from .encoder import Encoder
from .radial_mask import get_radial_mask
from .transformer import Transformer
from .utils.corresponding_edges import get_corresponding_edges
from .utils.edges_to_nef import edge_array_to_nef, get_nef_indices, nef_array_to_edges
from .utils.jax_batch import JAXBatch


class PET(eqx.Module):

    all_species: List[int] = eqx.static_field
    species_to_species_index: jnp.ndarray = eqx.static_field
    encoder: Encoder
    transformer: Transformer
    readout: eqx.nn.MLP
    composition_weights: jnp.ndarray = (
        eqx.static_field
    )  # TODO: check that this isn't being trained...
    gnn_transformers: List[Transformer]
    gnn_contractions: List[eqx.nn.Linear]

    def __init__(self, all_species, hypers, composition_weights, key):
        n_species = len(all_species)
        print("hello 2")

        # Handle species
        self.all_species = all_species
        self.species_to_species_index = jnp.full(
            (jnp.max(all_species) + 1,),
            -1,
            dtype=(jnp.int64 if jax.config.jax_enable_x64 else jnp.int32),
        )
        for i, species in enumerate(all_species):
            self.species_to_species_index = self.species_to_species_index.at[
                species
            ].set(i)
        print("Species indices:", self.species_to_species_index)
        print("Number of species:", n_species)

        key_enc, key_attn, key_readout, key_gnns = jax.random.split(key, 4)
        self.encoder = Encoder(n_species, hypers["d_pet"], key_enc)
        self.transformer = Transformer(
            hypers["d_pet"],
            4 * hypers["d_pet"],
            hypers["num_heads"],
            hypers["num_attention_layers"],
            hypers["mlp_dropout_rate"],
            hypers["attention_dropout_rate"],
            key_attn,
        )
        self.readout = eqx.nn.Linear(
            hypers["d_pet"], 1, use_bias=False, key=key_readout
        )
        num_mp_layers = hypers["num_gnn_layers"] - 1
        gnn_keys = jax.random.split(key_gnns, num_mp_layers)
        self.gnn_transformers = []
        self.gnn_contractions = []
        for i in range(num_mp_layers):
            contraction_key, transformer_key = jax.random.split(gnn_keys[i])
            self.gnn_contractions.append(
                eqx.nn.Linear(
                    2 * hypers["d_pet"],
                    hypers["d_pet"],
                    use_bias=False,
                    key=contraction_key,
                )
            )
            self.gnn_transformers.append(
                Transformer(
                    hypers["d_pet"],
                    4 * hypers["d_pet"],
                    hypers["num_heads"],
                    hypers["num_attention_layers"],
                    hypers["mlp_dropout_rate"],
                    hypers["attention_dropout_rate"],
                    transformer_key,
                )
            )

        self.composition_weights = composition_weights

    def __call__(self, structures, max_edges_per_node, is_training, key=None):

        n_structures = len(structures.n_nodes)

        # Convert to NEF:
        nef_indices, nef_to_edges_neighbor, nef_mask = get_nef_indices(
            structures.centers, len(structures.positions), max_edges_per_node
        )

        segment_indices = jnp.repeat(
            jnp.arange(n_structures),
            structures.n_nodes,
            total_repeat_length=len(structures.positions),
        )
        # segment_indices = segment_indices.at[len(structures.positions):]
        # .set(n_structures)

        # get edge vectors:
        edge_vectors = (
            structures.positions[structures.neighbors]
            - structures.positions[structures.centers]
            + jnp.einsum(
                "ia, iab -> ib",
                structures.cell_shifts,
                structures.cells[segment_indices[structures.centers]],
            )
        )

        # Get radial mask
        r = jnp.sqrt(jnp.sum(edge_vectors**2, axis=-1))
        radial_mask = jax.vmap(get_radial_mask, in_axes=(0, None, None))(r, 5.0, 3.0)

        # Element indices
        element_indices_nodes = self.species_to_species_index[structures.numbers]
        element_indices_centers = element_indices_nodes[structures.centers]
        element_indices_neighbors = element_indices_nodes[structures.neighbors]

        # Send everything to NEF:
        edge_vectors = edge_array_to_nef(edge_vectors, nef_indices)
        radial_mask = edge_array_to_nef(radial_mask, nef_indices, nef_mask, 0.0)
        element_indices_centers = edge_array_to_nef(
            element_indices_centers, nef_indices
        )
        element_indices_neighbors = edge_array_to_nef(
            element_indices_neighbors, nef_indices
        )

        features = {
            "cartesian": edge_vectors,
            "center": element_indices_centers,
            "neighbor": element_indices_neighbors,
        }

        # Encode
        features = self.encoder(features)

        # Transformer
        features = jax.vmap(self.transformer, in_axes=(0, None, 0, None))(
            features, is_training, radial_mask, key
        )

        # GNN
        num_mp_layers = len(self.gnn_transformers)
        if num_mp_layers > 0:
            corresponding_edges = get_corresponding_edges(
                jnp.stack([structures.centers, structures.neighbors], axis=-1)
            )
            for i in range(num_mp_layers):
                new_features = nef_array_to_edges(
                    features, structures.centers, nef_to_edges_neighbor
                )
                corresponding_new_features = new_features[corresponding_edges]
                new_features = jax.vmap(self.gnn_contractions[i])(
                    jnp.concatenate([new_features, corresponding_new_features], axis=-1)
                )
                new_features = edge_array_to_nef(new_features, nef_indices)
                new_features = jax.vmap(
                    self.gnn_transformers[i], in_axes=(0, None, 0, None)
                )(new_features, is_training, radial_mask, key)
                features = features + new_features

        # Readout
        edge_energies = jax.vmap(jax.vmap(self.readout))(features)
        edge_energies = edge_energies * radial_mask[:, :, None]

        # Sum over edges
        atomic_energies = jnp.sum(
            edge_energies, axis=(1, 2)
        )  # also eliminate singleton dimension 2

        # Sum over centers
        structure_energies = jax.ops.segment_sum(
            atomic_energies,
            segment_indices,
            num_segments=n_structures,
            indices_are_sorted=True,
        )

        # Add composition weights
        composition = jnp.empty((n_structures, len(self.all_species)))
        for number in self.all_species:
            where_number = (structures.numbers == number).astype(composition.dtype)
            composition = composition.at[:, self.species_to_species_index[number]].set(
                jax.ops.segment_sum(
                    where_number,
                    segment_indices,
                    num_segments=n_structures,
                )
            )

        structure_energies = structure_energies + composition @ self.composition_weights

        return {"energies": structure_energies}


@eqx.filter_grad
def predict_forces(
    positions: jax.Array,
    model: eqx.Module,
    structures: JAXBatch,
    max_edges_per_node,
    is_training,
    key,
):
    structures = structures._replace(positions=positions)
    return jnp.sum(model(structures, max_edges_per_node, is_training, key)["energies"])


class PET_energy_force(eqx.Module):

    pet: PET

    def __init__(self, all_species, hypers, composition_weights, key):
        print("hello 1")
        self.pet = PET(all_species, hypers, composition_weights, key)

    def __call__(self, structures, max_edges_per_node, is_training, key=None):
        energies = self.pet(structures, max_edges_per_node, is_training, key)[
            "energies"
        ]
        minus_forces = predict_forces(
            structures.positions,
            self.pet,
            structures,
            max_edges_per_node,
            is_training,
            key,
        )
        return {"energies": energies, "forces": -minus_forces}
