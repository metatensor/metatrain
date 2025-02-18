"""The main entry point for the metatrain command line interface."""

import argparse
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from . import PACKAGE_ROOT, __version__
from .cli.eval import _add_eval_model_parser, _prepare_eval_model_args, eval_model
from .cli.export import (
    _add_export_model_parser,
    _prepare_export_model_args,
    export_model,
)
from .cli.train import _add_train_model_parser, _prepare_train_model_args, train_model
from .utils.distributed.logging import is_main_process
from .utils.logging import get_cli_input, setup_logging


logger = logging.getLogger(__name__)


def _datetime_output_path(now: datetime) -> Path:
    """Get a date and time based output path."""
    return Path(
        "outputs",
        now.strftime("%Y-%m-%d"),
        now.strftime("%H-%M-%S"),
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    if len(sys.argv) < 2:
        ap.error("You must specify a sub-command")

    # If you change the synopsis of these commands or add new ones adjust the completion
    # script at `src/metatrain/share/metatrain-completion.bash`.
    ap.add_argument(
        "--version",
        action="version",
        version=f"metatrain {__version__}",
    )

    ap.add_argument(
        "--debug",
        action="store_true",
        help="Run with debug options.",
    )

    ap.add_argument(
        "--shell-completion",
        action="version",
        help="Path to the shell completion script",
        version=str(PACKAGE_ROOT / "share/metatrain-completion.bash"),
    )

    # Add sub-parsers
    subparser = ap.add_subparsers(help="sub-command help")
    _add_eval_model_parser(subparser)
    _add_export_model_parser(subparser)
    _add_train_model_parser(subparser)

    args = ap.parse_args()
    callable = args.__dict__.pop("callable")
    debug = args.__dict__.pop("debug")
    log_file = None
    error_file = Path("error.log")

    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if callable == "train_model":
        # define and create `checkpoint_dir` based on current directory, date and time
        checkpoint_dir = _datetime_output_path(now=datetime.now())
        if is_main_process():
            try:
                os.makedirs(checkpoint_dir)
            except FileExistsError:
                # directory already exists from a different run, add a suffix
                # (.1, .2, ...) to the directory name
                initial_checkpoint_dir = checkpoint_dir
                i = 1
                while True:
                    try:
                        checkpoint_dir = f"{initial_checkpoint_dir}.{i}"
                        os.makedirs(checkpoint_dir)
                        break
                    except FileExistsError:
                        i += 1
                checkpoint_dir = Path(checkpoint_dir)
        args.checkpoint_dir = checkpoint_dir

        log_file = checkpoint_dir / "train.log"
        error_file = checkpoint_dir / error_file

    with setup_logging(logger, log_file=log_file, level=level):
        logging.info(f"Package directory: {PACKAGE_ROOT}")
        logging.info(f"Working directory: {Path('.').absolute()}")
        logging.info(f"Metatrain version: {__version__}")
        logging.info(f"Executed command: {get_cli_input()}")

        try:
            if callable == "eval_model":
                _prepare_eval_model_args(args)
                eval_model(**args.__dict__)
            elif callable == "export_model":
                _prepare_export_model_args(args)
                export_model(**args.__dict__)
            elif callable == "train_model":
                _prepare_train_model_args(args)
                train_model(**args.__dict__)
            else:
                raise ValueError("internal error when selecting a sub-command")
        except Exception as err:
            logging.error(
                "If the error message below is unclear, please help us improve it by "
                "opening an issue at https://github.com/metatensor/metatrain/issues. "
                "When opening the issue, please include the full traceback log from "
                f"{str(error_file.absolute().resolve())!r}. Thank you!\n\n{err}"
            )

            with open(error_file, "w") as f:
                f.write(traceback.format_exc())

            sys.exit(1)


if __name__ == "__main__":
    main()
