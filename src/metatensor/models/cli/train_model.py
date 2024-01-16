import argparse
import importlib
import logging
import warnings
from pathlib import Path
from typing import Union

import hydra
import torch
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


def _resolve_single_str(config):
    if isinstance(config, str):
        return OmegaConf.create({"read_from": config})
    else:
        return config


def expand_dataset_config(conf: Union[str, DictConfig]) -> DictConfig:
    """Expand a short hand notation in a dataset config to actual format.

    Currently the config si not checked if all keys are known. Unknown keys can be added
    and will be ignored and not deleted."""

    conf_path = CONFIG_PATH / "dataset"
    base_conf_structures = OmegaConf.load(conf_path / "structures.yaml")
    base_conf_target = OmegaConf.load(conf_path / "targets.yaml")
    base_conf_gradients_avail = OmegaConf.load(conf_path / "gradients_avail.yaml")
    base_conf_gradient = OmegaConf.load(conf_path / "gradient.yaml")

    known_gradient_keys = list(base_conf_gradients_avail.keys())

    # merge confif to get default configs for energies and other target config.
    base_conf_target = OmegaConf.merge(base_conf_target, base_conf_gradients_avail)
    base_conf_energy = base_conf_target.copy()
    base_conf_energy["forces"] = base_conf_gradient.copy()
    base_conf_energy["stress"] = base_conf_gradient.copy()

    if isinstance(conf, str):
        read_from = conf
        conf = OmegaConf.create(
            {"structures": read_from, "targets": {"energy": read_from}}
        )

    if type(conf["structures"]) is str:
        conf["structures"] = _resolve_single_str(conf["structures"])

    conf["structures"] = OmegaConf.merge(base_conf_structures, conf["structures"])

    if conf["structures"]["file_format"] is None:
        conf["structures"]["file_format"] = Path(conf["structures"]["read_from"]).suffix

    for target_key, target in conf["targets"].items():
        if type(target) is str:
            target = _resolve_single_str(target)

        # Add default gradients "energy" target section
        if target_key == "energy":
            # For special case of the "energy" we add the section for force and stress
            # gradient by default
            target = OmegaConf.merge(base_conf_energy, target)
        else:
            target = OmegaConf.merge(base_conf_target, target)

        if target["key"] is None:
            target["key"] = target_key

        # Update DictConfig to allow for config node interpolation of variables like
        # "read_from"
        conf["targets"][target_key] = target

        # Check with respect to full config `conf` to avoid errors of node interpolation
        if conf["targets"][target_key]["file_format"] is None:
            conf["targets"][target_key]["file_format"] = Path(
                conf["targets"][target_key]["read_from"]
            ).suffix

        # merge and interpolate possible present gradients with default config
        for gradient_key, gradient_conf in conf["targets"][target_key].items():
            if gradient_key in known_gradient_keys:
                if gradient_key:
                    gradient_conf = base_conf_gradient.copy()
                elif type(gradient_key) is str:
                    gradient_conf = _resolve_single_str(gradient_conf)

                if isinstance(gradient_conf, DictConfig):
                    gradient_conf = OmegaConf.merge(base_conf_gradient, gradient_conf)

                    if gradient_conf["key"] is None:
                        gradient_conf["key"] = gradient_key

                    conf["targets"][target_key][gradient_key] = gradient_conf

        # If user sets the virial gradient and leaves the stress section untouched
        # we disable the by default enabled stress gradients.
        base_stress_gradient_conf = base_conf_gradient.copy()
        base_stress_gradient_conf["key"] = "stress"

        if (
            target_key == "energy"
            and conf["targets"][target_key]["virial"]
            and conf["targets"][target_key]["stress"] == base_stress_gradient_conf
        ):
            conf["targets"][target_key]["stress"] = False

        if (
            conf["targets"][target_key]["stress"]
            and conf["targets"][target_key]["virial"]
        ):
            raise ValueError(
                f"Cannot perform training with respect to virials and stress as in "
                f"section {target_key}. Set either `virials: off` or `stress: off`."
            )

    return conf


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

    # TODO load seed from config
    generator = torch.Generator()

    logger.info("Setting up training set")
    conf_training_set = expand_dataset_config(options["training_set"])
    structures_train = read_structures(
        filename=conf_training_set["structures"]["read_from"],
        fileformat=conf_training_set["structures"]["file_format"],
    )
    targets_train = read_targets(conf_training_set["targets"])
    train_dataset = Dataset(structures_train, targets_train)

    logger.info("Setting up test set")
    conf_test_set = options["test_set"]
    if not isinstance(conf_test_set, float):
        conf_test_set = expand_dataset_config(conf_test_set)
        structures_test = read_structures(
            filename=conf_training_set["structures"]["read_from"],
            fileformat=conf_training_set["structures"]["file_format"],
        )
        targets_test = read_targets(conf_test_set["targets"])
        test_dataset = Dataset(structures_test, targets_test)
        fraction_test_set = 0.0
    else:
        if conf_test_set < 0 or conf_test_set >= 1:
            raise ValueError("Test set split must be between 0 and 1.")
        fraction_test_set = conf_test_set

    logger.info("Setting up validation set")
    conf_validation_set = options["validation_set"]
    if not isinstance(conf_validation_set, float):
        conf_validation_set = expand_dataset_config(conf_validation_set)
        structures_validation = read_structures(
            filename=conf_training_set["structures"]["read_from"],
            fileformat=conf_training_set["structures"]["file_format"],
        )
        targets_validation = read_targets(conf_validation_set["targets"])
        validation_dataset = Dataset(structures_validation, targets_validation)
        fraction_validation_set = 0.0
    else:
        if conf_validation_set < 0 or conf_validation_set >= 1:
            raise ValueError("Validation set split must be between 0 and 1.")
        fraction_validation_set = conf_validation_set

    # Split train dataset if requested
    if fraction_test_set or fraction_validation_set:
        fraction_train_set = 1 - fraction_test_set - fraction_validation_set
        if fraction_train_set < 0:
            raise ValueError("fraction of the train set is smaller then 0!")

        # ignore warning of possible empty dataset
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            subsets = torch.utils.data.random_split(
                dataset=train_dataset,
                lengths=[
                    fraction_train_set,
                    fraction_test_set,
                    fraction_validation_set,
                ],
                generator=generator,
            )

        train_dataset = subsets[0]
        if fraction_test_set and not fraction_validation_set:
            test_dataset = subsets[1]
        elif not fraction_validation_set and fraction_validation_set:
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

    logger.info("Run training")
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # HACK: Avoid passing a Subset which we can not handle yet. We now
    if isinstance(train_dataset, torch.utils.data.Subset):
        model = architecture.train(
            train_dataset=train_dataset.dataset,
            hypers=OmegaConf.to_container(options["architecture"]),
            output_dir=output_dir,
        )
    else:
        model = architecture.train(
            train_dataset=train_dataset,
            hypers=OmegaConf.to_container(options["architecture"]),
            output_dir=output_dir,
        )

    save_model(model, options["output_path"])
