import jax
import jax.numpy as jnp
import numpy as np
import torch
from metatensor.torch.atomistic import ModelOutput

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

from . import DATASET_PATH


def test_pet_to_torch():
    """Tests that the model can be converted to torch and predict the same output."""

    all_species = [1, 6, 7, 8]
    hypers = {
        "d_pet": 128,
        "num_heads": 2,
        "num_attention_layers": 3,
        "mlp_dropout_rate": 0.0,
        "attention_dropout_rate": 0.0,
    }
    composition_weights = [0.1, 0.2, 0.3, 0.4]
    pet_jax = PET_jax(
        jnp.array(all_species), hypers, composition_weights, key=jax.random.PRNGKey(0)
    )

    systems = read_structures_ase(DATASET_PATH)
    systems = systems[:5]

    # jax evaluation
    jax_structures = [
        structure_to_jax(mts_to_structure(system, 0.0, np.zeros((0, 3)), 4.0))
        for system in systems
    ]
    jax_batch = jax_structures_to_batch(
        [structure_to_jax(structure) for structure in jax_structures]
    )
    _, _, n_edges_per_node = calculate_padding_sizes(jax_batch)
    output_jax = pet_jax(jax_batch, n_edges_per_node, is_training=False)

    # convert to torch
    pet_torch = pet_to_torch(pet_jax, hypers)

    # torch evaluation
    output_torch = pet_torch(systems, {"energy": ModelOutput()})

    assert torch.allclose(
        torch.tensor(output_jax["energy"]),
        output_torch["energy"].block().values.squeeze(-1),
    )
