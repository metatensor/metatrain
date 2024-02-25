import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from metatensor.torch.atomistic import ModelOutput, NeighborsListOptions

from metatensor.models.experimental.pet_jax.pet.models import PET as PET_jax
from metatensor.models.experimental.pet_jax.pet.utils.jax_batch import (
    calculate_padding_sizes,
    jax_structures_to_batch,
)
from metatensor.models.experimental.pet_jax.pet.utils.jax_structure import (
    structure_to_jax,
)
from metatensor.models.experimental.pet_jax.pet.utils.mts_to_structure import (
    mts_to_structure,
)
from metatensor.models.experimental.pet_jax.pet.utils.to_torch import pet_to_torch
from metatensor.models.utils.data.readers.structures import read_structures_ase
from metatensor.models.utils.neighbors_lists import get_system_with_neighbors_lists

from . import DATASET_PATH


def test_static_composition():
    """Checks that the composition features are not being trained."""

    all_species = [1, 6, 7, 8]
    hypers = {
        "d_pet": 128,
        "num_heads": 2,
        "num_attention_layers": 3,
        "num_gnn_layers": 1,
        "mlp_dropout_rate": 0.0,
        "attention_dropout_rate": 0.0,
    }
    composition_weights = jnp.array([0.1, 0.2, 0.3, 0.4])
    pet_jax = PET_jax(
        jnp.array(all_species),
        hypers,
        composition_weights,
        key=jax.random.PRNGKey(0),
    )

    systems = read_structures_ase(DATASET_PATH, dtype=torch.get_default_dtype())
    systems = systems[:5]
    jax_structures = [
        structure_to_jax(mts_to_structure(system, 0.0, np.zeros((0, 3)), 4.0))
        for system in systems
    ]
    jax_batch = jax_structures_to_batch(
        [structure_to_jax(structure) for structure in jax_structures]
    )
    _, _, n_edges_per_node = calculate_padding_sizes(jax_batch)

    def loss_fn(pet, batch, n_edges_per_node):
        output = pet(batch, n_edges_per_node, is_training=True)
        return jnp.sum((output["energies"] - jnp.zeros_like(output["energies"])) ** 2)

    grad_fn = eqx.filter_grad(loss_fn)
    gradients = grad_fn(pet_jax, jax_batch, n_edges_per_node)

    optimizer = optax.adam(learning_rate=1.0)
    optimizer_state = optimizer.init(eqx.filter(pet_jax, eqx.is_inexact_array))
    updates, optimizer_state = optimizer.update(gradients, optimizer_state, pet_jax)
    pet_jax = eqx.apply_updates(pet_jax, updates)

    print(pet_jax.composition_weights, composition_weights)
    assert jnp.allclose(pet_jax.composition_weights, composition_weights)
