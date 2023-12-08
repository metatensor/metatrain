"""The main entry point for the metatensor models interface."""
import argparse
import sys

from . import __version__
from .cli import eval_model, export_model, train_model


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument(
        "--version",
        action="version",
        version=f"metatensor-models {__version__}",
    )

    subparser = ap.add_subparsers(help="sub-command help")
    evaluate_parser = subparser.add_parser(
        "eval",
        help=eval_model.__doc__,
        description="eval model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    evaluate_parser.set_defaults(callable="eval_model")

    export_parser = subparser.add_parser(
        "export",
        help=export_model.__doc__,
        description="export model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    export_parser.set_defaults(callable="export_model")
    train_parser = subparser.add_parser(
        "train",
        help=train_model.__doc__,
        description="train model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        ap.parse_args([callable])
        if callable == "eval":
            eval_model()
        elif callable == "export":
            export_model()


if __name__ == "__main__":
    main()
