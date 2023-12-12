import argparse

from ..utils.data.readers import read_structures
from ..utils.data.writers import write_predictions
from ..utils.model_io import load_model

def _eval_model_cli(parser: argparse.ArgumentParser) -> None:
    """Add the `eval_model` paramaters to an argparse (sub)-parser"""
    parser.add_argument(
        "-m",
        "--model",
        dest="model_path",
        type=str,
        required=True,
        help="Path to a saved model",
    )
    parser.add_argument(
        "-s",
        "--structures",
        dest="structure_path",
        type=str,
        required=True,
        help="Path to a structure file which should be considered for the evaluation.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        type=str,
        required=False,
        default="output.xyz",
        help="Path to save the predicted values.",
    )


def eval_model(
    model_path: str, structure_path: str, output_path: str = "output.xyz"
) -> None:
    """Evaluate a pretrained model.

    ``target_property`` wil be predicted on a provided set of structures. Predicted
    values will be written ``output_path``.

    :param model_path: Path to a saved model
    :param structure_path: Path to a structure file which should be considered for the
        evaluation.
    :param output_path: Path to save the predicted values

    """

    model = load_model(model_path)
    structures = read_structures(structure_path)
    predictions = model(structures)
    write_predictions(output_path, predictions, structures)
