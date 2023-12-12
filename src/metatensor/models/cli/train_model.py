import importlib
import logging
import argparse
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

from metatensor.models.utils.data import Dataset
from metatensor.models.utils.data.readers import read_structures, read_targets
from ..utils.model_io import save_model

from .. import CONFIG_PATH


logger = logging.getLogger(__name__)

def _has_yaml_suffix(s: str) -> str:
    """Checks if a string has a .yaml suffix."""

    if Path(s).suffix != ".yaml":
        raise argparse.ArgumentTypeError(
            f"Parameters file '{s}' must be a `.yaml` file.")

    return s

def _train_model_cli(parser: argparse.ArgumentParser) -> None:
    """Add basic the `train_model` paramaters to an argparse (sub)-parser.

    This is just the first layer of arguments. Additional arguments are allowed and will
    be parsed by the hydra CLI."""
    parser.add_argument(
        "-p",
        "--parameters",
        dest="parameters_path",
        type=_has_yaml_suffix,
        required=True,
        help="Path to the parameter file",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        type=str,
        required=False,
        default="model.pt",
        help="Path to save the final model.",
    )
    parser.add_argument(
        "-y",
        "--hydra",
        dest="hydra_paramters",
        nargs="+",
        type=str,
        help="Flags for the hydra library",
    )


@hydra.main(config_path=str(CONFIG_PATH), config_name="config", version_base=None)
def train_model(config: DictConfig) -> None:
    """Train a model."""

    print(config)

    logger.info("Setting up dataset")
    structures = read_structures(config["dataset"]["structure_path"])
    targets = read_targets(
        config["dataset"]["targets_path"],
        target_values=config["dataset"]["target_value"],
    )
    dataset = Dataset(structures, targets)

    logger.info("Setting up model")
    architetcure_name = config["architecture"]["name"]
    architecture = importlib.import_module(f"metatensor.models.{architetcure_name}")
    model = architecture.Model(
        all_species=dataset.all_species,
        hypers=OmegaConf.to_container(config["architecture"]["model"]),
    )

    logger.info("Run training")
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    model = architecture.train(
        model=model,
        train_dataset=dataset,
        hypers=OmegaConf.to_container(config["architecture"]["training"]),
        output_dir=output_dir,
    )

    # Save the model:
    save_model(model, "model.pt")
