import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Union

import torch
from metatomic.torch import ModelMetadata, is_atomistic_model
from omegaconf import OmegaConf

from ..utils.io import check_file_extension, download_model_from_hf, load_model
from ..utils.metadata import merge_metadata
from .formatter import CustomHelpFormatter


def _add_export_model_parser(subparser: argparse._SubParsersAction) -> None:
    """Add `export_model` paramaters to an argparse (sub)-parser.

    :param subparser: The argparse (sub)-parser to add the parameters to.
    """

    if export_model.__doc__ is not None:
        description = export_model.__doc__.split(r":param")[0]
    else:
        description = None

    # If you change the synopsis of these commands or add new ones adjust the completion
    # script at `src/metatrain/share/metatrain-completion.bash`.
    parser = subparser.add_parser(
        "export",
        description=description,
        formatter_class=CustomHelpFormatter,
    )
    parser.set_defaults(callable="export_model")

    parser.add_argument(
        "path",
        type=str,
        nargs="+",
        help=(
            "Saved model which should be exported. Path can be either a URL, a "
            "local file, or a Hugging Face Hub identifier followed by the file name "
            "(e.g. 'metatensor/metatrain-test model.ckpt')."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        required=False,
        help=(
            "Filename of the exported model (default: <stem>.pt, "
            "where <stem> is the name of the checkpoint without the extension)."
        ),
    )
    parser.add_argument(
        "-e",
        "--extensions",
        dest="extensions",
        type=str,
        required=False,
        default="extensions/",
        help=(
            "Folder where the extensions of the model, if any, will be collected "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "-m",
        "--metadata",
        type=str,
        required=False,
        dest="metadata",
        default=None,
        help="Metatdata YAML file to be appended to the model.",
    )
    parser.add_argument(
        "-r",
        "--revision",
        "-b",
        "--branch",
        dest="revision",
        type=str,
        default=None,
        required=False,
        help="Revision (branch, tag, or commit) to download from Hugging Face.",
    )
    parser.add_argument(
        "--token",
        dest="hf_token",
        type=str,
        required=False,
        default=None,
        help="HuggingFace API token to download (private) models from HuggingFace. "
        "You can also set a environment variable `HF_TOKEN` to avoid passing it every "
        "time.",
    )


def _prepare_export_model_args(args: argparse.Namespace) -> None:
    """Prepare arguments for export_model.

    :param args: The argparse.Namespace containing the arguments.
    """

    hf_token = args.__dict__.get("hf_token", None)

    # use env variable if available
    env_hf_token = os.environ.get("HF_TOKEN")
    if env_hf_token:
        if hf_token is None:
            args.__dict__["hf_token"] = env_hf_token
        else:
            raise ValueError(
                "Both CLI and environment variable tokens are set for HuggingFace. "
                "Please use only one."
            )

    if args.metadata is not None:
        args.metadata = ModelMetadata(**OmegaConf.load(args.metadata))

    # Handle the nargs='+' for path
    path_args = args.path
    if len(path_args) == 1:
        args.path = path_args[0]
        args.path_in_repo = None
    elif len(path_args) == 2:
        args.path = path_args[0]
        args.path_in_repo = path_args[1]
    else:
        raise ValueError(
            "Too many arguments provided for 'path'."
            f" Expected 1 or 2, got {len(path_args)}: {path_args}"
        )

    # only these are needed for `export_model``
    keys_to_keep = [
        "path",
        "path_in_repo",
        "output",
        "extensions",
        "hf_token",
        "metadata",
        "revision",
    ]
    original_keys = list(args.__dict__.keys())

    for key in original_keys:
        if key not in keys_to_keep:
            args.__dict__.pop(key)

    # Logic to determine default output filename based on input source
    if args.__dict__.get("output") is None:
        if args.path_in_repo is not None:
            stem = Path(args.path_in_repo).stem
        else:
            stem = Path(args.path).stem

        args.__dict__["output"] = stem + ".pt"


def export_model(
    path: Union[Path, str],
    output: Union[Path, str],
    path_in_repo: Optional[str] = None,
    extensions: Union[Path, str] = "extensions/",
    hf_token: Optional[str] = None,
    metadata: Optional[ModelMetadata] = None,
    revision: Optional[str] = None,
) -> None:
    """Export a trained model allowing it to make predictions.

    This includes predictions within molecular simulation engines. Exported models will
    be saved with a ``.pt`` file ending. If ``path`` does not end with this file
    extensions ``.pt`` will be added and a warning emitted.

    The user can specify the model in three ways:

    1.  **Local File**:

        .. code-block:: bash

            mtt export model.ckpt

    2.  **Hugging Face Repository** (GitHub-style):

        .. code-block:: bash

            mtt export metatensor/metatrain-test model.ckpt

    3.  **Any URL**:

        .. code-block:: bash

            mtt export https://huggingface.co/metatensor/metatrain-test/resolve/main/model.ckpt

    :param path: path to a model file to be exported, or a Hugging Face repo ID
    :param output: path to save the model
    :param path_in_repo: path to the model file within the Hugging Face repository
    :param extensions: path to save the extensions
    :param hf_token: HuggingFace API token to download (private) models from HuggingFace
        (optional)
    :param metadata: metadata to be appended to the model
    :param revision: Revision (branch, tag, or commit) to download from Hugging Face
    """
    # Resolve Hugging Face repository path if applicable
    if path_in_repo is not None:
        logging.info(f"Downloading '{path_in_repo}' from '{path}'...")
        path = download_model_from_hf(
            repo_id=str(path),
            filename=path_in_repo,
            revision=revision,
            token=hf_token,
        )

    if Path(output).suffix == ".ckpt":
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")

        path = str(Path(output).absolute().resolve())
        extensions_path = None

        if metadata is not None:
            current_metadata = checkpoint.get("metadata", ModelMetadata())
            metadata = merge_metadata(current_metadata, metadata)
            checkpoint["metadata"] = metadata

        torch.save(checkpoint, path)
    else:
        # Here, we implicitly export the best_model_checkpoint
        # from the checkpoint path. See load_model code for details.
        model = load_model(path=path, hf_token=hf_token)
        path = str(
            Path(check_file_extension(filename=output, extension=".pt"))
            .absolute()
            .resolve()
        )

        if _has_extensions():
            extensions_path = str(Path(extensions).absolute().resolve())
        else:
            extensions_path = None

        if not is_atomistic_model(model):
            model = model.export(metadata)

        model.save(path, collect_extensions=extensions_path)
    if extensions_path is not None:
        logging.info(
            f"Model exported to '{path}' and extensions to '{extensions_path}'"
        )
    else:
        logging.info(f"Model exported to '{path}'")


def _has_extensions() -> bool:
    """
    Check if any torch extensions are currently loaded, except for metatensor_torch and
    metatomic_torch.

    :return: Whether extensions are loaded or not.
    """
    loaded_libraries = torch.ops.loaded_libraries
    for lib in loaded_libraries:
        if "metatensor_torch." in lib:
            continue
        elif "metatomic_torch." in lib:
            continue
        return True
    return False
