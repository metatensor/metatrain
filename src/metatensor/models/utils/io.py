import importlib
import os
import warnings
from pathlib import Path
from typing import Any, Union

import metatensor.torch
import torch
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelMetadata,
)


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


def check_suffix(filename: Union[str, Path], suffix: str) -> Union[str, Path]:
    """Check the suffix of a file name and adds if it not existing.

    If ``filename`` does not end with ``suffix`` the ``suffix`` is added and a warning
    will be issued.

    :param filename: Name of the file to be checked.
    :param suffix: Expected filesuffix i.e. ``.txt``.
    :returns: Checked and probably extended file name.
    """
    path_filename = Path(filename)

    if path_filename.suffix != suffix:
        warnings.warn(
            f"The file name should have a '{suffix}' extension. The user "
            f"requested the file with name '{filename}', but it will be saved as "
            f"'{filename}{suffix}'.",
            stacklevel=1,
        )
        path_filename = path_filename.parent / (path_filename.name + suffix)

    if type(filename) is str:
        return str(path_filename)
    else:
        return path_filename


def save(
    model: torch.nn.Module,
    path: Union[str, Path],
) -> None:
    """Saves a model to a checkpoint file.

    Along with the model all the metadata needed to load it is saved as well.
    Checkpointed models will be saved with a ``.ckpt`` file ending. If ``path`` does not
    end with this file extensions ``.ckpt`` will be added and a warning emitted.

    :param model: The model to save.
    :param path: The path to the file where the model should be saved.
    """

    if isinstance(path, Path):
        path = str(path)

    if not path.endswith(".ckpt"):
        path += ".ckpt"
        warnings.warn(
            message=f"adding '.ckpt' extension, the file will be saved at '{path}'",
            stacklevel=1,
        )

    torch.save(
        {
            "architecture_name": model.name,
            "model_state_dict": model.state_dict(),
            "model_hypers": model.hypers,
            "model_capabilities": model.capabilities,
        },
        path,
    )


def load(
    path: Union[str, Path]
) -> Union[torch.nn.Module, torch.jit._script.RecursiveScriptModule]:
    """Loads a checkpoint or an exported model from a file.

    :param path: path to load the model
    :return: the loaded model
    """

    if isinstance(path, Path):
        path = str(path)

    if not os.path.exists(path):
        raise ValueError(f"{path}: no such file or directory")
    elif path.endswith(".ckpt"):
        return _load_checkpoint(path)
    elif path.endswith(".pt"):
        return _load_exported_model(path)
    else:
        raise ValueError(
            f"{path} is neither a valid 'checkpoint' nor an 'exported' model"
        )


def _load_checkpoint(path: str) -> torch.nn.Module:
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


def _load_exported_model(path: str) -> torch.jit._script.RecursiveScriptModule:
    metatensor.torch.atomistic.check_atomistic_model(path)
    return torch.jit.load(path)


def export(model: torch.nn.Module, path: Union[str, Path]) -> None:
    """Export a trained model to allow it to make predictions.

    This includes predictions within molecular simulation engines. Exported models will
    be saved with a ``.pt`` file ending. If ``path`` does not end with this file
    extensions ``.pt`` will be added and a warning emitted.

    :param model: model to be exported
    :param path: path to save the exported model
    """

    if isinstance(path, Path):
        path = str(path)

    if not path.endswith(".pt"):
        path += ".pt"
        warnings.warn(
            message=f"adding '.pt' extension, the file will be saved at '{path}'",
            stacklevel=1,
        )

    # if the model is already exported we just save it again to the given path
    if is_exported(model):
        torch.jit.save(model, path)
        return

    if model.capabilities.length_unit == "":
        warnings.warn(
            "No `length_unit` was provided for the model. As a result, lengths "
            "and any derived quantities will be passed to MD engines as is.",
            stacklevel=1,
        )

    for model_output_name, model_output in model.capabilities.outputs.items():
        if model_output.unit == "":
            warnings.warn(
                f"No target units were provided for output {model_output_name!r}. "
                "As a result, this model output will be passed to MD engines as is.",
                stacklevel=1,
            )

    model_capabilities_with_devices = ModelCapabilities(
        length_unit=model.capabilities.length_unit,
        atomic_types=model.capabilities.atomic_types,
        outputs=model.capabilities.outputs,
        supported_devices=["cpu", "cuda"],
    )

    wrapper = MetatensorAtomisticModel(
        model.eval(), ModelMetadata(), model_capabilities_with_devices
    )
    wrapper.export(path)


def is_exported(model: Any):
    """Check if a model has been exported.

    :param model: The model to check
    :return: :py:obj:`True` if the ``model`` has been exported, :py:obj:`False`
        otherwise
    """
    return isinstance(model, torch.jit._script.RecursiveScriptModule)
