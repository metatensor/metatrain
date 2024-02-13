import argparse
import importlib
import logging
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import hydra
import numpy as np
import torch
from metatensor.learn.data import Dataset
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigKeyError

from .. import CONFIG_PATH
from ..utils.data import get_all_species, read_structures, read_targets
from ..utils.data.dataset import _train_test_random_split
from ..utils.model_io import save_model
from ..utils.omegaconf import check_units, expand_dataset_config
from .eval_model import _eval_targets
from .formatter import CustomHelpFormatter


logger = logging.getLogger(__name__)


def _add_train_model_parser(subparser: argparse._SubParsersAction) -> None:
    """Add basic the `train_model` paramaters to an argparse (sub)-parser.

    This is just the first layer of arguments. Additional arguments are allowed and will
    be parsed by the hydra CLI."""

    if train_model.__doc__ is not None:
        description = train_model.__doc__.split(r":param")[0]
    else:
        description = None

    parser = subparser.add_parser(
        "train",
        description=description,
        formatter_class=CustomHelpFormatter,
    )
    parser.set_defaults(callable="train_model")

    parser.add_argument(
        "options",
        type=OmegaConf.load,
        help="Options file",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        required=False,
        default="model.pt",
        help="Path to save the final model (default: %(default)s).",
    )
    parser.add_argument(
        "-c",
        "--continue",
        dest="continue_from",
        type=str,
        required=False,
        help="File to continue training from.",
    )
    parser.add_argument(
        "-y",
        "--hydra",
        dest="hydra_parameters",
        nargs="+",
        type=str,
        help="Hydra's command line and override flags.",
    )


def train_model(
    options: DictConfig,
    output: str = "model.pt",
    continue_from: Optional[str] = None,
    hydra_parameters: Optional[List[str]] = None,
) -> None:
    """
    Train an atomistic machine learning model using configurations provided by Hydra.

    This function sets up the dataset and model architecture, then runs the training
    process. The dataset is prepared by reading structural data and target values from
    specified paths. The model architecture is dynamically imported and instantiated
    based on the configuration. Training is executed with the specified hyperparameters,
    and the trained model is saved to a designated output path.

    Hydra is used for command-line configuration management, allowing for dynamic
    parameter setting at runtime. See
    https://hydra.cc/docs/advanced/hydra-command-line-flags/ and
    https://hydra.cc/docs/advanced/override_grammar/basic/ for details.

    :param options: DictConfig containing the training options
    :param output: Path to save the final model
    :param continue_from: File to continue training from.
    :param hydra_parameters: Hydra's command line and override flags
    """
    try:
        architecture_name = options["architecture"]["name"]
    except ConfigKeyError as exc:
        raise ConfigKeyError("Architecture name is not defined!") from exc

    options["defaults"] = [
        "base",
        {"architecture": architecture_name},
        {"override hydra/job_logging": "custom"},
        "_self_",
    ]

    # HACK: Hydra parses command line arguments directlty from `sys.argv`. We override
    # `sys.argv` and write files to a tempory directory to be hydra compatible with our
    # CLI architecture.
    with tempfile.TemporaryDirectory() as tmpdirname:
        options_new = Path(tmpdirname) / "options.yaml"
        OmegaConf.save(config=options, f=options_new)

        if continue_from is None:
            continue_from = "null"

        argv = sys.argv[:1]
        argv.append(f"--config-dir={options_new.parent}")
        argv.append(f"--config-name={options_new.name}")
        argv.append(f"+output_path={output}")
        argv.append(f"+continue_from={continue_from}")

        if hydra_parameters is not None:
            argv += hydra_parameters

        sys.argv = argv

        _train_model_hydra()


@hydra.main(config_path=str(CONFIG_PATH), version_base=None)
def _train_model_hydra(options: DictConfig) -> None:
    """Actual fit function called in :func:`train_model`.

    :param options: A dictionary-like object obtained from Hydra, containing all the
        necessary options for dataset preparation, model hyperparameters, and training.
    """
    if options["base_precision"] == 64:
        torch.set_default_dtype(torch.float64)
    elif options["base_precision"] == 32:
        torch.set_default_dtype(torch.float32)
    elif options["base_precision"] == 16:
        torch.set_default_dtype(torch.float16)
    else:
        raise ValueError("Only 64, 32 or 16 are possible values for `base_precision`.")

    if options["seed"] is not None:
        if options["seed"] < 0:
            raise ValueError("`seed` should be a positive number or None.")
        else:
            torch.manual_seed(options["seed"])
            np.random.seed(options["seed"])
            random.seed(options["seed"])
            os.environ["PYTHONHASHSEED"] = str(options["seed"])
            if torch.cuda.is_available():
                torch.cuda.manual_seed(options["seed"])
                torch.cuda.manual_seed_all(options["seed"])

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.info(f"This log is also available in '{output_dir}/train.log'.")

    logger.info("Setting up training set")
    train_options = expand_dataset_config(options["training_set"])
    train_structures = read_structures(
        filename=train_options["structures"]["read_from"],
        fileformat=train_options["structures"]["file_format"],
    )
    train_targets = read_targets(train_options["targets"])
    train_dataset = Dataset(structure=train_structures, energy=train_targets["energy"])
    train_size = 1.0

    logger.info("Setting up test set")
    test_options = options["test_set"]

    if isinstance(test_options, float):
        test_size = test_options
        train_size -= test_size

        if test_size < 0 or test_size >= 1:
            raise ValueError("Test set split must be between 0 and 1.")

        generator = torch.Generator()
        if options["seed"] is not None:
            generator.manual_seed(options["seed"])

        train_dataset, test_dataset = _train_test_random_split(
            train_dataset=train_dataset,
            train_size=train_size,
            test_size=test_size,
            generator=generator,
        )
    else:
        test_options = expand_dataset_config(test_options)
        test_structures = read_structures(
            filename=test_options["structures"]["read_from"],
            fileformat=test_options["structures"]["file_format"],
        )
        test_targets = read_targets(test_options["targets"])
        test_dataset = Dataset(structure=test_structures, energy=test_targets["energy"])
        check_units(actual_options=test_options, desired_options=train_options)

    logger.info("Setting up validation set")
    validation_options = options["validation_set"]
    if isinstance(validation_options, float):
        validation_size = validation_options
        train_size -= validation_size

        if validation_size < 0 or validation_size >= 1:
            raise ValueError("Validation set split must be between 0 and 1.")

        generator = torch.Generator()
        if options["seed"] is not None:
            generator.manual_seed(options["seed"])

        train_dataset, validation_dataset = _train_test_random_split(
            train_dataset=train_dataset,
            train_size=train_size,
            test_size=validation_size,
            generator=generator,
        )
    else:
        validation_options = expand_dataset_config(validation_options)
        validation_structures = read_structures(
            filename=validation_options["structures"]["read_from"],
            fileformat=validation_options["structures"]["file_format"],
        )
        validation_targets = read_targets(validation_options["targets"])
        validation_dataset = Dataset(
            structure=validation_structures, energy=validation_targets["energy"]
        )
        check_units(actual_options=validation_options, desired_options=train_options)

    # Save fully expanded config
    OmegaConf.save(config=options, f=Path(output_dir) / "options.yaml")

    logger.info("Setting up model")
    architetcure_name = options["architecture"]["name"]
    architecture = importlib.import_module(f"metatensor.models.{architetcure_name}")

    all_species = []
    for dataset in [train_dataset]:  # HACK: only a single train_dataset for now
        all_species += get_all_species(dataset)
    all_species = list(set(all_species))
    all_species.sort()

    outputs = {
        key: ModelOutput(
            quantity=value["quantity"],
            unit=(value["unit"] if value["unit"] is not None else ""),
        )
        for key, value in options["training_set"]["targets"].items()
    }
    length_unit = train_options["structures"]["length_unit"]
    requested_capabilities = ModelCapabilities(
        length_unit=length_unit if length_unit is not None else "",
        species=all_species,
        outputs=outputs,
    )

    logger.info("Calling architecture trainer")
    model = architecture.train(
        train_datasets=[train_dataset],
        validation_datasets=[validation_dataset],
        requested_capabilities=requested_capabilities,
        hypers=OmegaConf.to_container(options["architecture"]),
        continue_from=options["continue_from"],
        output_dir=output_dir,
        device_str=options["device"],
    )

    save_model(model, options["output_path"])

    logger.info("Evaulate train dataset")
    _eval_targets(model, train_dataset)

    logger.info("Evaulate validation dataset")
    _eval_targets(model, validation_dataset)

    logger.info("Evaulate test dataset")
    _eval_targets(model, test_dataset)
