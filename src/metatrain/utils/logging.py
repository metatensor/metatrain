"""Logging."""

import contextlib
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from metatensor.torch.atomistic import ModelOutput

from .external_naming import to_external_name
from .io import check_suffix


class MetricLogger:
    def __init__(
        self,
        logobj: logging.Logger,
        model_outputs: Dict[str, ModelOutput],
        initial_metrics: Union[Dict[str, float], List[Dict[str, float]]],
        names: Union[str, List[str]] = "",
    ):
        """
        Simple interface to log training metrics logging instance.

        Initialize the metric logger. The logger is initialized with the initial metrics
        and names relative to the metrics (e.g., "train", "validation").

        In this way, and by assuming that these metrics never increase, the logger can
        align the output to make it easier to read.

        :param logobj: A logging instance
        :param model_outputs: outputs of the model
        :param initial_metrics: initial training metrics
        :param validation_info_0: initial validation metrics
        """
        self.logobj = logobj
        self.model_outputs = model_outputs

        if isinstance(initial_metrics, dict):
            initial_metrics = [initial_metrics]
        if isinstance(names, str):
            names = [names]

        self.names = names

        # Since the quantities are supposed to decrease, we want to store the
        # number of digits at the start of the training, so that we can align
        # the output later:
        self.digits = {}
        for name, metrics_dict in zip(names, initial_metrics):
            for key, value in metrics_dict.items():
                if "loss" in key:
                    # losses will be printed in scientific notation
                    continue
                self.digits[f"{name}_{key}"] = _get_digits(value)

        # Save the model outputs. This will be useful to know
        # what physical quantities we are printing
        self.model_outputs = model_outputs

    def log(
        self,
        metrics: Union[Dict[str, float], List[Dict[str, float]]],
        epoch: Optional[int] = None,
    ):
        """
        Log the metrics.

        The metrics are automatically aligned to make them easier to read, based on
        the order of magnitude of each metric given to the class at initialization.

        :param metrics: The current metrics to be printed.
        :param epoch: The current epoch (optional). If :py:class:`None`, the epoch
            will not be printed, and the logging string will start with the first
            metric in the ``metrics`` dictionary.
        """

        if isinstance(metrics, dict):
            metrics = [metrics]

        if epoch is None:
            logging_string = ""
        else:
            # The epoch is printed with 4 digits, assuming that the training
            # will not last more than 9999 epochs
            logging_string = f"Epoch {epoch:4}"

        for name, metrics_dict in zip(self.names, metrics):
            for key, value in metrics_dict.items():

                new_key = key
                if key != "loss":  # special case: not a metric associated with a target
                    target_name, metric = new_key.split(" ", 1)
                    target_name = to_external_name(target_name, self.model_outputs)
                    new_key = f"{target_name} {metric}"

                if name == "":
                    logging_string += f", {new_key}: "
                else:
                    logging_string += f", {name} {new_key}: "
                if "loss" in key:  # print losses with scientific notation
                    logging_string += f"{value:.3e}"
                else:
                    logging_string += f"{value:{self.digits[f'{name}_{key}'][0]}.{self.digits[f'{name}_{key}'][1]}f}"  # noqa: E501

        # If there is no epoch, the string will start with a comma. Remove it:
        if logging_string.startswith(", "):
            logging_string = logging_string[2:]

        self.logobj.info(logging_string)


def _get_digits(value: float) -> Tuple[int, int]:
    """
    Finds the number of digits to print before and after the decimal point,
    based on the order of magnitude of the value.

    5 "significant" digits are guaranteed to be printed.

    :param value: The value for which the number of digits is calculated.
    """

    # Get order of magnitude of the value:
    order = int(np.floor(np.log10(value)))

    # Get the number of digits before the decimal point:
    if order < 0:
        digits_before = 1
    else:
        digits_before = order + 1

    # Get the number of digits after the decimal point:
    if order < 0:
        digits_after = 4 - order
    else:
        digits_after = max(1, 4 - order)

    total_characters = digits_before + digits_after + 1  # +1 for the point

    return total_characters, digits_after


@contextlib.contextmanager
def setup_logging(
    log_obj: logging.Logger,
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.WARNING,
):
    """Create a logging environment for a given ``logobj``.

    Extracted and adjusted from
    github.com/MDAnalysis/mdacli/blob/main/src/mdacli/logger.py

    :param log_obj: A logging instance
    :param log_file: Name of the log file
    :param level: Set the root logger level to the specified level. If for example set
        to :py:obj:`logging.DEBUG` detailed debug logs inludcing filename and function
        name are displayed. For :py:obj:`logging.INFO` only the message logged from
        `errors`, `warnings` and `infos` will be displayed.
    """
    try:
        format = "[{asctime}][{levelname}]"
        if level == logging.DEBUG:
            format += ":{filename}:{funcName}:{lineno}"
        format += " - {message}"

        formatter = logging.Formatter(format, datefmt="%Y-%m-%d %H:%M:%S", style="{")
        handlers: List[Union[logging.StreamHandler, logging.FileHandler]] = []

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        handlers.append(stream_handler)

        if log_file:
            log_file = check_suffix(filename=log_file, suffix=".log")
            file_handler = logging.FileHandler(filename=str(log_file), encoding="utf-8")
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)

        logging.basicConfig(format=format, handlers=handlers, level=level, style="{")

        if log_file:
            abs_path = str(Path(log_file).absolute().resolve())
            log_obj.info(f"This log is also available at {abs_path!r}.")
        else:
            log_obj.info("Logging to file is disabled.")

        for handler in handlers:
            log_obj.addHandler(handler)

        yield

    finally:
        for handler in handlers:
            handler.flush()
            handler.close()
            log_obj.removeHandler(handler)
