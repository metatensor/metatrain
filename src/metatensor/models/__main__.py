"""The main entry point for the metatensor models interface."""
import argparse
import sys

from . import __version__
from .cli import eval_model, export_model, train_model
from .cli.eval_model import _eval_model_cli


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument(
        "--version",
        action="version",
        version=f"metatensor-models {__version__}",
    )

    subparser = ap.add_subparsers(help="sub-command help")
    evaluate_parser = subparser.add_parser(
        "eval",
        help="Evaluate a pretrained model.",
        description="eval model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    evaluate_parser.set_defaults(callable="eval_model")
    _eval_model_cli(evaluate_parser)

    export_parser = subparser.add_parser(
        "export",
        help=export_model.__doc__,
        description="export model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    export_parser.set_defaults(callable="export_model")
    train_parser = subparser.add_parser(
        "train",
        help=train_model.__doc__,
        description="train model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_parser.set_defaults(callable="train_model")

    if len(sys.argv) < 2:
        ap.error("You must specify a sub-command")

    # Be case insensitive for the subcommand
    sys.argv[1] = sys.argv[1].lower()
    callable = sys.argv[1].lower()

    # Workaround since we are using hydra for the train parsing
    if callable == "train":
        # Remove "metatensor_model" command to please hydra
        sys.argv.pop(0)
        train_model()

    else:
        args = ap.parse_args()

        # Remove `callable`` because it is not an argument of the two functions
        args.__dict__.pop("callable")
        if callable == "eval":
            eval_model(**args.__dict__)
        elif callable == "export":
            export_model(**args.__dict__)


if __name__ == "__main__":
    main()
