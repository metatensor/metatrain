import jax
import torch

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
        all_species=list(pet_jax.all_species),
        hypers=hypers,
        composition_weights=torch.tensor(
            pet_jax.composition_weights.numpy(), device=device
        ),
    )

    # skip the species list (in both atomic numbers indices) and composition weights
    jax_params = [x for x in jax.tree_util.tree_leaves() if isinstance(x, jax.Array)][
        2:-1
    ]
    torch_params = list(pet_torch.parameters())

    for jax_param, torch_param in zip(jax_params, torch_params):
        torch_param.data = torch.tensor(jax_param.numpy(), device=device)

    return pet_torch
