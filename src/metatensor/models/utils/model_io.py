import importlib
from pathlib import Path
from typing import Union
import warnings

import torch


def save_model(
    model: torch.nn.Module,
    path: Union[str, Path],
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
            "model_capabilities": model.capabilities,
        },
        path,
    )


def load_model(path: Union[str, Path]) -> torch.nn.Module:
    """Loads a model from a file.

    Parameters
    ----------
    :param path: The path to the file containing the model.

    Returns
    -------
    :return: The loaded model.
    """

    if isinstance(path, str):
        path = Path(path)

    if path.suffix == ".pt":
        warnings.warn(
            "Trying to load a checkpoint from a .pt file. Unless you renamed "
            "it, this is probably an exported model which will fail to train.",
            stacklevel=1,
        )

    # Load the model and the metadata
    model_dict = torch.load(path)

    # Get the architecture
    architecture = importlib.import_module(
        f"metatensor.models.{model_dict['architecture_name']}"
    )

    # Create the model
    model = architecture.Model(
        capabilities=model_dict["model_capabilities"], hypers=model_dict["model_hypers"]
    )

    # Load the model weights
    model.load_state_dict(model_dict["model_state_dict"])

    return model
