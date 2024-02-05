import argparse
import importlib
import logging
import warnings
from pathlib import Path

import hydra
import torch
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput
from omegaconf import DictConfig, OmegaConf

from metatensor.models.utils.data import Dataset
from metatensor.models.utils.data.readers import read_structures, read_targets

from .. import CONFIG_PATH
from ..utils.data import get_all_species
from ..utils.model_io import save_model
from ..utils.omegaconf import expand_dataset_config
from .formatter import CustomHelpFormatter


logger = logging.getLogger(__name__)


def _has_yaml_suffix(s: str) -> str:
    """Checks if a string has a .yaml suffix."""

    if Path(s).suffix != ".yaml":
        raise argparse.ArgumentTypeError(f"Options file '{s}' must be a `.yaml` file.")

    return s


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
        type=_has_yaml_suffix,
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
        default=None,
        help="File to continue training from.",
    )
    parser.add_argument(
        "-y",
        "--hydra",
        dest="hydra_paramters",
        nargs="+",
        type=str,
        help="Hydra's command line and override flags.",
    )


@hydra.main(config_path=str(CONFIG_PATH), version_base=None)
def train_model(options: DictConfig) -> None:
    """Train an atomistic machine learning model using configurations provided by Hydra.

    This function sets up the dataset and model architecture, then runs the training
    process. The dataset is prepared by reading structural data and target values from
    specified paths. The model architecture is dynamically imported and instantiated
    based on the configuration. Training is executed with the specified hyperparameters,
    and the trained model is saved to a designated output path.

    Hydra is used for command-line configuration management, allowing for dynamic
    parameter setting at runtime. See
    https://hydra.cc/docs/advanced/hydra-command-line-flags/ and
    https://hydra.cc/docs/advanced/override_grammar/basic/ for details.

    :param options: A dictionary-like object obtained from Hydra, containing all the
        necessary options for dataset preparation, model hyperparameters, and training.
    """

    # This gives some accuracy improvements. It is very likely that
    # this is just due to the preliminary composition fit in the SOAP-BPNN.
    # TODO: investigate
    torch.set_default_dtype(torch.float64)

    # TODO load seed from config
    generator = torch.Generator()

    logger.info("Setting up training set")
    train_options = expand_dataset_config(options["training_set"])
    train_structures = read_structures(
        filename=train_options["structures"]["read_from"],
        fileformat=train_options["structures"]["file_format"],
    )
    train_targets = read_targets(train_options["targets"])
    train_dataset = Dataset(train_structures, train_targets)

    logger.info("Setting up test set")
    test_options = options["test_set"]
    if not isinstance(test_options, float):
        test_options = expand_dataset_config(test_options)
        test_structures = read_structures(
            filename=test_options["structures"]["read_from"],
            fileformat=test_options["structures"]["file_format"],
        )
        test_targets = read_targets(test_options["targets"])
        test_dataset = Dataset(test_structures, test_targets)
        test_fraction = 0.0
    else:
        if test_options < 0 or test_options >= 1:
            raise ValueError("Test set split must be between 0 and 1.")
        test_fraction = test_options

    logger.info("Setting up validation set")
    validation_options = options["validation_set"]
    if not isinstance(validation_options, float):
        validation_options = expand_dataset_config(validation_options)
        validation_structures = read_structures(
            filename=validation_options["structures"]["read_from"],
            fileformat=validation_options["structures"]["file_format"],
        )
        validation_targets = read_targets(validation_options["targets"])
        validation_dataset = Dataset(validation_structures, validation_targets)
        validation_fraction = 0.0
    else:
        if validation_options < 0 or validation_options >= 1:
            raise ValueError("Validation set split must be between 0 and 1.")
        validation_fraction = validation_options

    # Split train dataset if requested
    if test_fraction or validation_fraction:
        train_fraction = 1 - test_fraction - validation_fraction
        if train_fraction < 0:
            raise ValueError("fraction of the train set is smaller then 0!")

        # ignore warning of possible empty dataset
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            subsets = torch.utils.data.random_split(
                dataset=train_dataset,
                lengths=[
                    train_fraction,
                    test_fraction,
                    validation_fraction,
                ],
                generator=generator,
            )

        train_dataset = subsets[0]
        if test_fraction and not validation_fraction:
            test_dataset = subsets[1]
        elif not validation_fraction and validation_fraction:
            validation_dataset = subsets[1]
        else:
            test_dataset = subsets[1]
            validation_dataset = subsets[2]

    # TODO: Perform section and unit consistency checks between test/train/validation
    # set
    test_dataset
    validation_dataset

    logger.info("Setting up model")
    architetcure_name = options["architecture"]["name"]
    architecture = importlib.import_module(f"metatensor.models.{architetcure_name}")

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

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
    model_capabilities = ModelCapabilities(
        length_unit=length_unit if length_unit is not None else "",
        species=all_species,
        outputs=outputs,
    )

    logger.info("Calling architecture trainer")
    model = architecture.train(
        train_datasets=[train_dataset],
        validation_datasets=[validation_dataset],
        model_capabilities=model_capabilities,
        hypers=OmegaConf.to_container(options["architecture"]),
        continue_from=options["continue"],
        output_dir=output_dir,
    )

    save_model(model, options["output_path"])

    # TODO: add evaluation of the test set
