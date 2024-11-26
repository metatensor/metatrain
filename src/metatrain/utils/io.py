import logging
import os
import shutil
import warnings
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urlparse
from urllib.request import urlretrieve

import torch
from metatensor.torch.atomistic import check_atomistic_model, load_atomistic_model

from ..utils.architectures import find_all_architectures
from .architectures import import_architecture


logger = logging.getLogger(__name__)


def check_file_extension(
    filename: Union[str, Path], extension: str
) -> Union[str, Path]:
    """Check the file extension of a file name and adds if it is not present.

    If ``filename`` does not end with ``extension`` the ``extension`` is added and a
    warning will be issued.

    :param filename: Name of the file to be checked.
    :param extension: Expected file extension i.e. ``.txt``.
    :returns: Checked and probably extended file name.
    """
    path_filename = Path(filename)

    if path_filename.suffix != extension:
        warnings.warn(
            f"The file name should have a '{extension}' file extension. The user "
            f"requested the file with name '{filename}', but it will be saved as "
            f"'{filename}{extension}'.",
            stacklevel=1,
        )
        path_filename = path_filename.parent / (path_filename.name + extension)

    if type(filename) is str:
        return str(path_filename)
    else:
        return path_filename


def is_exported_file(path: str) -> bool:
    """
    Check if a saved model file has been exported to a ``MetatensorAtomisticModel``.

    The functions uses :py:func:`metatensor.torch.atomistic.check_atomistic_model` to
    verify.

    :param path: model path
    :return: :py:obj:`True` if the ``model`` has been exported, :py:obj:`False`
        otherwise.

    .. seealso::

        :py:func:`metatensor.torch.atomistic.is_atomistic_model` to verify if an already
        loaded model is exported.
    """
    try:
        check_atomistic_model(str(path))
        return True
    except ValueError:
        return False


def load_model(
    path: Union[str, Path],
    extensions_directory: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Any:
    """Load checkpoints and exported models from an URL or a local file.

    If an exported model should be loaded and requires compiled extensions, their
    location should be passed using the ``extensions_directory`` parameter.

    After reading a checkpoint, the returned
    model can be exported with the model's own ``export()`` method.

    :param path: local or remote path to a model. For supported URL schemes see
        :py:class`urllib.request`
    :param extensions_directory: path to a directory containing all extensions required
        by an *exported* model

    :raises ValueError: if ``path`` is a YAML option file and no model
    :raises ValueError: if no ``archietcture_name`` is found in the checkpoint
    :raises ValueError: if the ``architecture_name`` is not found in the available
        architectures
    """

    if Path(path).suffix in [".yaml", ".yml"]:
        raise ValueError(
            f"path '{path}' seems to be a YAML option file and not a model"
        )

    # Download from HuggingFace with a private token
    if (
        kwargs.get("huggingface_api_token")  # token from CLI
        or os.environ.get("HF_TOKEN")  # token from env variable
    ) and "huggingface.co" in str(path):
        cli_token = kwargs.get("huggingface_api_token")
        env_token = os.environ.get("HF_TOKEN")
        if cli_token and env_token:
            logging.info(
                "Both CLI and environment variable tokens are set for "
                "HuggingFace. Using the CLI token."
            )
            hf_token = cli_token
        else:
            if cli_token:
                hf_token = cli_token
            if env_token:
                hf_token = env_token
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "To download a model from HuggingFace, please install the "
                "`huggingface_hub` package with pip (`pip install "
                "huggingface_hub`)."
            )
        path = str(path)
        if not path.startswith("https://huggingface.co/"):
            raise ValueError(
                f"Invalid URL '{path}'. HuggingFace models should start with "
                "'https://huggingface.co/'."
            )
        # get repo_id and filename
        split_path = path.split("/")
        repo_id = f"{split_path[3]}/{split_path[4]}"  # org/repo
        filename = ""
        for i in range(5, len(split_path)):
            filename += split_path[i] + "/"
        filename = filename[:-1]
        if filename.startswith("resolve"):
            if not filename[8:].startswith("main/"):
                raise ValueError(
                    f"Invalid URL '{path}'. metatrain only supports models from the "
                    "'main' branch."
                )
            filename = filename[13:]
        if filename.startswith("blob/"):
            if not filename[5:].startswith("main/"):
                raise ValueError(
                    f"Invalid URL '{path}'. metatrain only supports models from the "
                    "'main' branch."
                )
            filename = filename[10:]
        path = hf_hub_download(repo_id, filename, token=hf_token)
        # make sure to copy the checkpoint to the current directory
        basename = os.path.basename(path)
        shutil.copy(path, Path.cwd() / basename)
        logger.info(f"Downloaded model from HuggingFace to {basename}")

    elif urlparse(str(path)).scheme:
        path, _ = urlretrieve(str(path))
        # make sure to copy the checkpoint to the current directory
        basename = os.path.basename(path)
        shutil.copy(path, Path.cwd() / basename)
        logger.info(f"Downloaded model to {basename}")

    else:
        pass

    path = str(path)
    if is_exported_file(path):
        return load_atomistic_model(path, extensions_directory=extensions_directory)
    else:  # model is a checkpoint
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        if "architecture_name" not in checkpoint:
            raise ValueError("No architecture name found in the checkpoint")
        architecture_name = checkpoint["architecture_name"]
        if architecture_name not in find_all_architectures():
            raise ValueError(
                f"Checkpoint architecture '{architecture_name}' not found "
                "in the available architectures. Available architectures are: "
                f"{find_all_architectures()}"
            )
        architecture = import_architecture(architecture_name)

        try:
            return architecture.__model__.load_checkpoint(path)
        except Exception as err:
            raise ValueError(
                f"path '{path}' is not a valid checkpoint for the {architecture_name} "
                "architecture"
            ) from err
