import argparse

from ..utils.export import export
from ..utils.model_io import load_checkpoint
from .formatter import CustomHelpFormatter


def _add_export_model_parser(subparser: argparse._SubParsersAction) -> None:
    if export.__doc__ is not None:
        description = export.__doc__.split(":param")[0]
    else:
        description = None

    parser = subparser.add_parser(
        "export",
        description=description,
        formatter_class=CustomHelpFormatter,
    )
    parser.set_defaults(callable="export_model")

    parser.add_argument(
        "model",
        type=load_checkpoint,
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
