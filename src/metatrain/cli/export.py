import argparse
from pathlib import Path
from typing import Union

import torch

from ..utils.architectures import find_all_architectures
from ..utils.export import is_exported
from ..utils.io import check_suffix
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
        "architecture_name",
        type=str,
        choices=find_all_architectures(),
        help="name of the model's architecture",
    )
    parser.add_argument(
        "path",
        type=str,
        help="Saved model which should be exported",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        required=False,
        default="exported-model.pt",
        help="Filename of the exported model (default: %(default)s).",
    )


def export_model(
    model: torch.nn.Module, output: Union[Path, str] = "exported-model.pt"
) -> None:
    """Export a trained model to allow it to make predictions.

    This includes predictions within molecular simulation engines. Exported models will
    be saved with a ``.pt`` file ending. If ``path`` does not end with this file
    extensions ``.pt`` will be added and a warning emitted.

    :param model: model to be exported
    :param output: path to save the exported model
    """
    path = str(check_suffix(filename=output, suffix=".pt"))

    if is_exported(model):
        torch.jit.save(model, path)
    else:
        exported_model = export(model)
        exported_model.export(path)
