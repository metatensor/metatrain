import argparse

from ..utils.data.readers import read_structures
from ..utils.data.writers import write_predictions
from ..utils.model_io import load_model
from .formatter import CustomHelpFormatter


def _add_eval_model_parser(subparser: argparse._SubParsersAction) -> None:
    """Add the `eval_model` paramaters to an argparse (sub)-parser"""

    if eval_model.__doc__ is not None:
        description = eval_model.__doc__.split(r":param")[0]
    else:
        description = None

    parser = subparser.add_parser(
        "eval",
        description=description,
        formatter_class=CustomHelpFormatter,
    )
    parser.set_defaults(callable="eval_model")

    parser.add_argument(
        "model",
        type=str,
        help="saved model to be evaluated",
    )
    parser.add_argument(
        "structures",
        type=str,
        help="Structure file which should be considered for the evaluation.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        required=False,
        default="output.xyz",
        help="filenmae of the predictions (default: %(default)s)",
    )


def eval_model(model: str, structures: str, output: str = "output.xyz") -> None:
    """Evaluate a pretrained model.

    ``target_property`` wil be predicted on a provided set of structures. Predicted
    values will be written ``output``.

    :param model: Path to a saved model
    :param structure: Path to a structure file which should be considered for the
        evaluation.
    :param output: Path to save the predicted values
    """

    loaded_model = load_model(model)
    structure_list = read_structures(structures)
    predictions = loaded_model(structure_list)
    write_predictions(output, predictions, structure_list)
