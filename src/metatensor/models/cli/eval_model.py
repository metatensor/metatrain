import argparse
import logging
from typing import Union
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ..utils.compute_loss import compute_model_loss
from ..utils.data import Dataset, read_structures, read_targets, write_predictions
from ..utils.loss import TensorMapDictLoss
from ..utils.model_io import load_model
from ..utils.omegaconf import check_units, expand_dataset_config, _has_yaml_suffix
from .formatter import CustomHelpFormatter
from ..utils.data.dataset import _train_test_random_split


logger = logging.getLogger(__name__)

CHOICES_EVAL_ON = ["train", "validation", "test"]


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
        "options",
        type=_has_yaml_suffix,
        help="Options file to define a test dataset taken for the evaluation.",
    )
    parser.add_argument(
        "model",
        type=str,
        help="saved model to be evaluated",
    )
    parser.add_argument(
        "eval_on",
        type=str,
        choices=CHOICES_EVAL_ON,
        help="On which part of the dataset should the model be evaluated.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        required=False,
        default="output.xyz",
        help="filename of the predictions (default: %(default)s)",
    )


def _eval_targets(model, dataset: Union[Dataset, torch.utils.data.Subset]):
    pass
    # weights_dict = {}  # logic to be taken from bpnn..
    # loss = TensorMapDictLoss(weights_dict)

    # targets = None
    # _, info = compute_model_loss(loss, model, dataset, targets)

    # TODO do the actual printing of RMSEs!


def eval_model(
    options: str, model: str, eval_on: str = "test", output: str = "output.xyz"
) -> None:
    """Evaluate a pretrained model on a set.

    The test dataset will be selected as defined in the options yaml file. Predicted
    values will be written ``output``.

    :param options: Options file path to define a test dataset taken for the evaluation.
    :param model: Path to a saved model
    :param eval_on: On which part of the dataset should the model be evaluated. Possible
        values are 'test', 'train' or 'validation'.
    :param output: Path to save the predicted values
    """

    conf = OmegaConf.load(options)

    if eval_on not in CHOICES_EVAL_ON:
        raise ValueError(
            f"{eval_on!r} is not a possible choice for `eval_on`. Choose from: "
            f"{','.join(CHOICES_EVAL_ON)}"
        )

    logger.info("Setting up {eval_on} set")
    train_options = conf["test_set"]
    eval_options = conf["{eval_on}_set"]

    loaded_model = load_model(model)

    eval_structures = read_structures(
        filename=eval_options["structures"]["read_from"],
        fileformat=eval_options["structures"]["file_format"],
    )

    # Predict targets
    if hasattr(eval_options, "targets"):
        if isinstance(eval_options, float):
            eval_size = eval_options
            train_size = 1 - eval_size

            if eval_size < 0 or eval_size >= 1:
                raise ValueError(f"{eval_on} set split must be between 0 and 1.")

            train_structures = read_structures(
                filename=train_options["structures"]["read_from"],
                fileformat=train_options["structures"]["file_format"],
            )
            train_targets = read_targets(train_options["targets"])
            train_dataset = Dataset(train_structures, train_targets)

            generator = torch.Generator()
            if options["seed"] is not None:
                generator.manual_seed(options["seed"])

            _, eval_dataset = _train_test_random_split(
                train_dataset=train_dataset,
                train_size=train_size,
                test_size=eval_size,
                generator=generator,
            )

            # Select eval_structures based on fraction
            eval_structures = [eval_structures[index] for index in eval_dataset.indices]

        else:
            eval_options = expand_dataset_config(eval_options)
            eval_targets = read_targets(eval_options["targets"])
            eval_dataset = Dataset(eval_structures, eval_targets)
            check_units(actual_options=eval_options, desired_options=train_options)

        _eval_targets(loaded_model, eval_dataset)

    # Predict strcutures
    predictions = loaded_model(eval_structures, loaded_model.capabilities.outputs)
    write_predictions(output, predictions, eval_structures)
