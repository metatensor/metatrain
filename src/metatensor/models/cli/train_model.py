import importlib
import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from metatensor.models.utils.data import Dataset
from metatensor.models.utils.data.readers import read_structures, read_targets

from .. import CONFIG_PATH


logger = logging.getLogger(__name__)


@hydra.main(config_path=str(CONFIG_PATH), config_name="config", version_base=None)
def train_model(config: DictConfig) -> None:
    """train a model."""

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
    logger.info(f"Changing directory for train output to: {output_dir}")
    os.chdir(output_dir)

    architecture.train(
        model=model,
        train_dataset=dataset,
        hypers=OmegaConf.to_container(config["architecture"]["training"]),
    )
