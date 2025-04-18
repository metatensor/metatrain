import re
import warnings
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import unquote, urlparse
from urllib.request import urlretrieve

import torch
from metatensor.torch.atomistic import check_atomistic_model, load_atomistic_model

from ..utils.architectures import find_all_architectures
from .architectures import import_architecture


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


def _hf_hub_download_url(url: str, token: Optional[str] = None) -> str:
    """Wrapper around `hf_hub_download` allowing passing the URL directly.

    Function is in inverse of `hf_hub_url`
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "To download a private model please install the `huggingface_hub` package "
            "with pip (`pip install huggingface_hub`)."
        )

    pattern = re.compile(
        r"(?P<endpoint>https://[^/]+)/"
        r"(?P<repo_id>[^/]+/[^/]+)/"
        r"resolve/"
        r"(?P<revision>[^/]+)/"
        r"(?P<filename>.+)"
    )

    match = pattern.match(url)

    if not match:
        raise ValueError(f"URL '{url}' has an invalid format for the Hugging Face Hub.")

    endpoint = match.group("endpoint")
    repo_id = match.group("repo_id")
    revision = unquote(match.group("revision"))
    filename = unquote(match.group("filename"))

    # Extract subfolder if applicable
    parts = filename.split("/", 1)
    if len(parts) == 2:
        subfolder, filename = parts
    else:
        subfolder = None

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
        repo_type=None,
        revision=revision,
        endpoint=endpoint,
        token=token,
    )


def load_model(
    path: Union[str, Path],
    extensions_directory: Optional[Union[str, Path]] = None,
    token: Optional[str] = None,
) -> Any:
    """Load checkpoints and exported models from an URL or a local file.

    If an exported model should be loaded and requires compiled extensions, their
    location should be passed using the ``extensions_directory`` parameter.

    After reading a checkpoint, the returned model can be exported with the model's own
    ``export()`` method.

    :param path: local or remote path to a model. For supported URL schemes see
        :py:class:`urllib.request`
    :param extensions_directory: path to a directory containing all extensions required
        by an *exported* model
    :param token: HuggingFace API token to download (private) models from HuggingFace

    :raises ValueError: if ``path`` is a YAML option file and no model
    :raises ValueError: if no ``archietcture_name`` is found in the checkpoint
    :raises ValueError: if the ``architecture_name`` is not found in the available
        architectures
    """

    if Path(path).suffix in [".yaml", ".yml"]:
        raise ValueError(
            f"path '{path}' seems to be a YAML option file and not a model"
        )

    path = str(path)

    # Download remote model
    # TODO(@PicoCentauri): Introduce caching for remote models
    if urlparse(path).scheme:
        if token is None:
            path, _ = urlretrieve(path)
        else:
            path = _hf_hub_download_url(path, token=token)

    if is_exported_file(path):
        return load_atomistic_model(path, extensions_directory=extensions_directory)
    else:  # model is a checkpoint
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")

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
