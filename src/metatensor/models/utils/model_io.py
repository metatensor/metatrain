import importlib
from typing import Dict, List

import torch


def save_model(
    arch_name: str,
    model: torch.nn.Module,
    hypers: Dict,
    all_species: List[int],
    path: str,
) -> None:
    """Saves a model to a file, along with all the metadata needed to load it.

    Parameters
    ----------
        arch_name (str): The name of the architecture.

        model (torch.nn.Module): The model to save.

        hypers (Dict): The hyperparameters used to train the model.

        all_species (List[int]): The list of all species that the model can handle.

        path (str): The path to the file.
    """
    torch.save(
        {
            "name": arch_name,
            "model": model.state_dict(),
            "hypers": hypers,
            "all_species": all_species,
        },
        path,
    )


def load_model(path: str) -> torch.nn.Module:
    """Loads a model from a file.

    Parameters
    ----------
        path (str): The path to the file.

    Returns
    -------
        torch.nn.Module: The loaded model.
    """

    # Load the model and the metadata
    model_dict = torch.load(path)

    # Get the architecture
    architecture = importlib.import_module(f"metatensor.models.{model_dict['name']}")

    # Create the model
    model = architecture.Model(
        all_species=model_dict["all_species"], hypers=model_dict["hypers"]
    )

    # Load the model weights
    model.load_state_dict(model_dict["model"])

    return model
