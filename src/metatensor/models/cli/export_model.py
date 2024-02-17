import argparse

from metatensor.torch.atomistic import MetatensorAtomisticModel

from ..utils.export import export
from ..utils.model_io import load_model
from .formatter import CustomHelpFormatter


def _add_export_model_parser(subparser: argparse._SubParsersAction) -> None:
    if export_model.__doc__ is not None:
        description = export_model.__doc__.split(":param")[0]
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
        type=load_model,
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


def export_model(model, output):
    """Exports a trained model to allow it to make predictions,
    including within molecular simulation engines.

    :param model: Path to a saved model checkpoint
    :param output: Path to save the exported model
    """

    export(model, output)
