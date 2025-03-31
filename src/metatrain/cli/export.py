import argparse
import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

from metatensor.torch.atomistic import ModelMetadata, is_atomistic_model
from omegaconf import OmegaConf

from ..utils.io import check_file_extension, load_model
from .formatter import CustomHelpFormatter


def _add_export_model_parser(subparser: argparse._SubParsersAction) -> None:
    """Add `export_model` paramaters to an argparse (sub)-parser."""

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
        help=(
            "Saved model which should be exported. Path can be either a URL or a "
            "local file."
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
        "-m",
        "--metadata",
        type=str,
        required=False,
        dest="metadata",
        default=None,
        help="Metatdata YAML file to be appended to the model.",
    )
    parser.add_argument(
        "--token",
        dest="token",
        type=str,
        required=False,
        default=None,
        help="HuggingFace API token to download (private )models from HuggingFace. "
        "You can also set a environment variable `HF_TOKEN` to avoid passing it every "
        "time.",
    )


def _prepare_export_model_args(args: argparse.Namespace) -> None:
    """Prepare arguments for export_model."""

    path = args.__dict__.pop("path")
    token = args.__dict__.pop("token")

    # use env variable if available
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        if token is None:
            token = env_token
        else:
            raise ValueError(
                "Both CLI and environment variable tokens are set for HuggingFace. "
                "Please use only one."
            )

    args.model = load_model(path=path, token=token)

    if args.metadata is not None:
        args.metadata = ModelMetadata(**OmegaConf.load(args.metadata))

    # only these are needed for `export_model``
    keys_to_keep = ["model", "output", "metadata"]
    original_keys = list(args.__dict__.keys())

    for key in original_keys:
        if key not in keys_to_keep:
            args.__dict__.pop(key)
    if args.__dict__.get("output") is None:
        args.__dict__["output"] = Path(path).stem + ".pt"


def export_model(
    model: Any, output: Union[Path, str], metadata: Optional[ModelMetadata] = None
) -> None:
    """Export a trained model allowing it to make predictions.

    This includes predictions within molecular simulation engines. Exported models will
    be saved with a ``.pt`` file ending. If ``path`` does not end with this file
    extensions ``.pt`` will be added and a warning emitted.

    :param model: model to be exported
    :param output: path to save the model
    :param metadata: metadata to be appended to the model
    """
    path = str(
        Path(check_file_extension(filename=output, extension=".pt"))
        .absolute()
        .resolve()
    )
    extensions_path = str(Path("extensions/").absolute().resolve())

    if not is_atomistic_model(model):
        model = model.export(metadata)

    model.save(path, collect_extensions=extensions_path)
    logging.info(f"Model exported to '{path}' and extensions to '{extensions_path}'")
