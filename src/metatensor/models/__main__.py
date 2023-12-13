"""The main entry point for the metatensor-models interface."""
import argparse
import sys
from pathlib import Path

from . import __version__
from .cli import eval_model, export_model, train_model
from .cli.eval_model import _add_eval_model_parser
from .cli.export_model import _add_export_model_parser
from .cli.train_model import _add_train_model_parser


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

    if len(sys.argv) < 2:
        ap.error("You must specify a sub-command")

    # Add sub-parsers
    subparser = ap.add_subparsers(help="sub-command help")
    _add_eval_model_parser(subparser)
    _add_export_model_parser(subparser)
    _add_train_model_parser(subparser)

    args = ap.parse_args()
    callable = args.__dict__.pop("callable")

    if callable == "eval_model":
        eval_model(**args.__dict__)
    elif callable == "export_model":
        export_model(**args.__dict__)
    elif callable == "train_model":
        # HACK: Hydra parses command line arguments directlty from `sys.argv`. We
        # override `sys.argv` to be compatible with our CLI architecture.
        argv = sys.argv[:1]

        parameters_path = Path(args.parameters_path)
        argv.append(f"--config-dir={parameters_path.parent}")
        argv.append(f"--config-name={parameters_path.name}")
        argv.append(f"+output_path={args.output_path}")

        if args.hydra_paramters is not None:
            argv += args.hydra_paramters

        sys.argv = argv

        train_model()
    else:
        raise ValueError("internal error when selecting a sub-command.")


if __name__ == "__main__":
    main()
