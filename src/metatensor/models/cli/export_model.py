import argparse

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
        type=str,
        help="Saved model which should be exprted",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        required=False,
        default="exported.pt",
        help="Filename of the exported model (default: %(default)s).",
    )


def export_model(model: str, output: str) -> None:
    """Export a pretrained model to run MD simulations

    :param model: Path to a saved model
    :param output: Path to save the exported model
    """
    raise NotImplementedError("model exporting is not implemented yet.")
