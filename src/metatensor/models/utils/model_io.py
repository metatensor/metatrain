import importlib
import warnings
from pathlib import Path
from typing import Union

import torch


# This import is necessary to avoid errors when loading an
# exported alchemical model, which depends on sphericart-torch.
# TODO: Remove this when https://github.com/lab-cosmo/metatensor/issues/512
# is ready
try:
    import sphericart.torch  # noqa: F401
except ImportError:
    pass

try:
    import rascaline.torch  # noqa: F401
except ImportError:
    pass


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


def load_checkpoint(path: Union[str, Path]) -> torch.nn.Module:
    """Loads a checkpoint from a file.

    :param path: The path to the file containing the checkpoint.

    :return: The loaded model.
    """

    if isinstance(path, str):
        path = Path(path)

    if path.suffix != ".ckpt":
        warnings.warn(
            f"Trying to load a checkpoint from a {path.suffix} file. This is probably "
            "not a checkpoint and will fail to train or export. Please use a `.ckpt` "
            "(checkpoint) file instead.",
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


def load_exported_model(
    path: Union[str, Path]
) -> torch.jit._script.RecursiveScriptModule:
    """Loads an exported model from a file.

    :param path: The path to the file containing the exported model.

    :return: The loaded model.
    """

    if isinstance(path, str):
        path = Path(path)

    if path.suffix != ".pt":
        warnings.warn(
            f"Trying to load an exported model from a {path.suffix} file. This is "
            "probably not an exported model, and it will fail to load. If this is a "
            "checkpoint (.ckpt), please export the checkpoint with the "
            "`metatensor-models export` command.",
            stacklevel=1,
        )

    try:
        return torch.jit.load(path)
    except RuntimeError as e:
        raise ValueError(f"model at {path} is not exported") from e
