"""The main entry point for the metatensor models interface."""
import argparse
import sys
from pathlib import Path
from . import __version__
from .cli import eval_model, export_model, train_model
from .cli.eval_model import _eval_model_cli
from .cli.train_model import _train_model_cli


def main():
    if len(sys.argv) < 2:
        ap.error("You must specify a sub-command")

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

    # Add train sub-parser
    train_parser = subparser.add_parser(
        "train",
        description=train_model.__doc__.split(r"\n:param")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_parser.set_defaults(callable="train_model")
    _train_model_cli(train_parser)

    # Add eval sub-parser
    evaluate_parser = subparser.add_parser(
        "eval",
        description=eval_model.__doc__.split(r"\n:param")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    evaluate_parser.set_defaults(callable="eval_model")
    _eval_model_cli(evaluate_parser)

    # Add export sub-parser
    export_parser = subparser.add_parser(
        "export",
        description=export_model.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    export_parser.set_defaults(callable="export_model")

    args = ap.parse_args()
    callable = args.__dict__.pop("callable")

    # Workaround since we are using hydra for the train parsing
    if callable == "train_model":

        # HACK: Hydra directly takes arguments from `sys.argv`.
        # We have to overwrite `sys.argv` to be compatible with our CLI architecture
        argv = sys.argv[:1]

        parameters_path = Path(args.parameters_path)
        argv.append(f"--config-dir={parameters_path.parent}")
        argv.append(f"--config-name={parameters_path.name}")
        #argv.append(f"+output_path={args.output_path}")

        if args.hydra_paramters is not None:
             argv += args.hydra_paramters

        sys.argv = argv
        print(sys.argv)

        train_model()
    if callable == "eval_model":
        eval_model(**args.__dict__)
    elif callable == "export_model":
        export_model(**args.__dict__)
    else:
        raise ValueError("internal error when setting up sub-commands.")


if __name__ == "__main__":
    main()
