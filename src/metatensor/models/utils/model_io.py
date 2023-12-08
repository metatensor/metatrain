import importlib

import torch


def save_model(
    model: torch.nn.Module,
    path: str,
) -> None:
    """Saves a model to a file, along with all the metadata needed to load it.

    Parameters
    ----------
    :param model: The model to save.
    :param path: The path to the file where the model should be saved.
    """
    torch.save(
        {
            "architecture_name": model.name,
            "model_state_dict": model.state_dict(),
            "model_hypers": model.hypers,
            "all_species": model.all_species,
        },
        path,
    )


def load_model(path: str) -> torch.nn.Module:
    """Loads a model from a file.

    Parameters
    ----------
    :param path: The path to the file containing the model.

    Returns
    -------
    :return: The loaded model.
    """

    # Load the model and the metadata
    model_dict = torch.load(path)

    # Get the architecture
    architecture = importlib.import_module(
        f"metatensor.models.{model_dict['architecture_name']}"
    )

    # Create the model
    model = architecture.Model(
        all_species=model_dict["all_species"], hypers=model_dict["model_hypers"]
    )

    # Load the model weights
    model.load_state_dict(model_dict["model_state_dict"])

    return model
