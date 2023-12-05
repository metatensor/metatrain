"""The main entry point for the metatensor model interface."""
import argparse
import sys

from . import __version__
from .scripts import evaluate, export, train


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

    ap.add_argument(
        "--debug",
        action="store_true",
        help="Run with debug options.",
    )

    ap.add_argument(
        "--logfile", dest="logfile", action="store", help="Logfile (optional)"
    )

    subparser = ap.add_subparsers(help="sub-command help")
    evaluate_parser = subparser.add_parser(
        "evaluate",
        help=evaluate.__doc__,
        description="evaluate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    evaluate_parser.set_defaults(callable="evaluate")

    export_parser = subparser.add_parser(
        "export",
        help=export.__doc__,
        description="export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    export_parser.set_defaults(callable="export")
    train_parser = subparser.add_parser(
        "train",
        help=train.__doc__,
        description="train",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    train_parser.set_defaults(callable="train")

    if len(sys.argv) < 2:
        ap.error("A subcommand is required.")

    # Be case insensitive for the subcommand
    sys.argv[1] = sys.argv[1].lower()

    args = ap.parse_args(sys.argv[1:])

    if args.callable == "evaluate":
        evaluate()
    elif args.callable == "export":
        export()
    elif args.callable == "train":
        train()


if __name__ == "__main__":
    main()
