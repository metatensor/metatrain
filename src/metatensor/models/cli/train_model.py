import argparse
import importlib
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from metatensor.models.utils.data import Dataset
from metatensor.models.utils.data.readers import read_structures, read_targets

from .. import CONFIG_PATH
from ..utils.model_io import save_model
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
        "-y",
        "--hydra",
        dest="hydra_paramters",
        nargs="+",
        type=str,
        help="Hydra's command line and override flags.",
    )


@hydra.main(config_path=str(CONFIG_PATH), config_name="config", version_base=None)
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

    logger.info("Setting up dataset")
    structures = read_structures(options["dataset"]["structure_path"])
    targets = read_targets(
        options["dataset"]["targets_path"],
        target_values=options["dataset"]["target_value"],
    )
    dataset = Dataset(structures, targets)

    logger.info("Setting up model")
    architetcure_name = options["architecture"]["name"]
    architecture = importlib.import_module(f"metatensor.models.{architetcure_name}")
    model = architecture.Model(
        all_species=dataset.all_species,
        hypers=OmegaConf.to_container(options["architecture"]["model"]),
    )

    logger.info("Run training")
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    print(OmegaConf.to_container(options))
    model = architecture.train(
        model=model,
        train_dataset=dataset,
        hypers=OmegaConf.to_container(options["architecture"]["training"]),
        output_dir=output_dir,
    )

    save_model(model, options["output_path"])
