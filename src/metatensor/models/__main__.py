"""The main entry point for the metatensor-models command line interface."""

import argparse
import sys
import traceback

from . import __version__
from .cli.eval import _add_eval_model_parser, eval_model
from .cli.export import _add_export_model_parser
from .cli.train import _add_train_model_parser, train_model
from .utils.export import export


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    if len(sys.argv) < 2:
        ap.error("You must specify a sub-command")

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

    # Add sub-parsers
    subparser = ap.add_subparsers(help="sub-command help")
    _add_eval_model_parser(subparser)
    _add_export_model_parser(subparser)
    _add_train_model_parser(subparser)

    args = ap.parse_args()
    callable = args.__dict__.pop("callable")
    debug = args.__dict__.pop("debug")

    try:
        if callable == "eval_model":
            eval_model(**args.__dict__)
        elif callable == "export_model":
            export(**args.__dict__)
        elif callable == "train_model":
            train_model(**args.__dict__)
        else:
            raise ValueError("internal error when selecting a sub-command.")
    except Exception as e:
        if debug:
            traceback.print_exc()
        else:
            sys.exit(f"\033[31mERROR: {e}\033[0m")  # format error in red!


if __name__ == "__main__":
    main()
