import jax
import torch
import numpy as np
import jax.numpy as jnp

from ...model import Model as PET_torch
from ..models import PET as PET_jax


def pet_to_torch(pet_jax: PET_jax, hypers: dict):
    """Convert a pet-jax model to a pet-torch model"""

    jax_device = pet_jax.composition_weights.device_buffer.device()
    if jax_device.platform == "cpu":
        torch_device_type = "cpu"
    elif jax_device.platform == "gpu":
        torch_device_type = "cuda"
    else:
        raise ValueError(
            f"Failed to convert device {jax_device.platform} "
            "during jax-to-torch conversion of PET-JAX"
        )
    device = torch.device(torch_device_type)

    pet_torch = PET_torch(
        all_species=torch.tensor(np.array(pet_jax.all_species), device=device),
        hypers=hypers,
        composition_weights=torch.tensor(
            np.array(pet_jax.composition_weights), device=device
        ),
    )

    # skip the species list (in both atomic numbers indices) and composition weights
    jax_params = [x for x in jax.tree_util.tree_leaves(pet_jax) if isinstance(x, jax.Array)][
        2:-1
    ]
    torch_params = list(pet_torch.parameters())

    torch_counter = 0
    jax_counter = 0
    while True:
        torch_param = torch_params[torch_counter]
        jax_param = jax_params[jax_counter]

        if torch_param.shape != jax_param.shape:
            if torch_param.shape[0] == 3*jax_param.shape[0] and torch_param.shape[1:] == jax_param.shape[1:]:
                # we're dealing with the attention weights
                jax_param = [jax_param]
                jax_param.append(jax_params[jax_counter+1])
                jax_param.append(jax_params[jax_counter+2])
                jax_counter += 2
                jax_param = jnp.concatenate(jax_param)
            else:
                raise ValueError(
                    f"Failed to convert parameter {torch_param.shape} "
                    f"to {jax_param.shape} during jax-to-torch conversion of PET-JAX"
                )
        torch_param.data = torch.tensor(np.array(jax_param), device=device)
        jax_counter += 1
        torch_counter += 1
        if jax_counter == len(jax_params):
            assert torch_counter == len(torch_params)
            break

    return pet_torch
