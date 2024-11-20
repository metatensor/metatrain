import warnings
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urlparse
from urllib.request import urlretrieve

from metatensor.torch.atomistic import check_atomistic_model, load_atomistic_model

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


def load_model(
    path: Union[str, Path],
    extensions_directory: Optional[Union[str, Path]] = None,
    architecture_name: Optional[str] = None,
) -> Any:
    """Load checkpoints and exported models from an URL or a local file.

    If an exported model should be loaded and requires compiled extensions, their
    location should be passed using the ``extensions_directory`` parameter.

    Loading checkpoints requires the ``architecture_name`` parameter, which can be
    ommited for loading an exported model. After reading a checkpoint, the returned
    model can be exported with the model's own ``export()`` method.

    :param path: local or remote path to a model. For supported URL schemes see
        :py:class`urllib.request`
    :param extensions_directory: path to a directory containing all extensions required
        by an *exported* model
    :param architecture_name: name of the architecture required for loading from a
        *checkpoint*.

    :raises ValueError: if both an ``extensions_directory`` and ``architecture_name``
        are given
    :raises ValueError: if ``path`` is a YAML option file and no model
    :raises ValueError: if no ``archietcture_name`` is given for loading a checkpoint
    :raises ValueError: if the checkpoint saved in ``path`` does not math the given
        ``architecture_name``
    """
    if extensions_directory is not None and architecture_name is not None:
        raise ValueError(
            f"Both ``extensions_directory`` ('{str(extensions_directory)}') and "
            f"``architecture_name`` ('{architecture_name}') are given which are "
            "mutually exclusive. An ``extensions_directory`` is only required for "
            "*exported* models while an ``architecture_name`` is only needed for model "
            "*checkpoints*."
        )

    if Path(path).suffix in [".yaml", ".yml"]:
        raise ValueError(f"path '{path}' seems to be a YAML option file and no model")

    if urlparse(str(path)).scheme:
        path, _ = urlretrieve(str(path))

    if is_exported_file(str(path)):
        return load_atomistic_model(
            str(path), extensions_directory=extensions_directory
        )
    else:  # model is a checkpoint
        if architecture_name is None:
            raise ValueError(
                f"path '{path}' seems to be a checkpointed model but no "
                "`architecture_name` was given"
            )
        architecture = import_architecture(architecture_name)

        try:
            return architecture.__model__.load_checkpoint(str(path))
        except Exception as err:
            raise ValueError(
                f"path '{path}' is not a valid model file for the {architecture_name} "
                "architecture"
            ) from err
