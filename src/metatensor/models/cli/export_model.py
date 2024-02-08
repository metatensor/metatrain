import argparse
import warnings

import torch
from metatensor.torch.atomistic import MetatensorAtomisticModel

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


def export_model(model: torch.nn.Module, output: str) -> None:
    """Export a pre-trained model to run MD simulations

    :param model: Path to a saved model
    :param output: Path to save the exported model
    """

    for model_output_name, model_output in model.capabilities.outputs.items():
        if model_output.unit == "":
            warnings.warn(
                f"No units were provided for the `{model_output_name}` output. "
                "As a result, this model output will be passed to MD engines as is.",
                stacklevel=1,
            )

    wrapper = MetatensorAtomisticModel(model.eval(), model.capabilities)
    wrapper.export(output)
