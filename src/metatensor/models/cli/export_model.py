import argparse


def _add_export_model_parser(subparser: argparse._SubParsersAction) -> None:
    if export_model.__doc__ is not None:
        description = export_model.__doc__.split(r":param")[0]
    else:
        description = None

    parser = subparser.add_parser(
        "export",
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(callable="export_model")

    parser.add_argument(
        "-m",
        "--model",
        dest="model_path",
        type=str,
        required=True,
        help="Path to a saved model",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        type=str,
        required=False,
        default="exported.pt",
        help="Export path for the model.",
    )


def export_model(model_path: str, output_path: str) -> None:
    """Export a pretrained model to run MD simulations

    :param model_path: Path to a saved model
    :param output_path: Path to save the exported model
    """
    raise NotImplementedError("model exporting is not implemented yet.")
