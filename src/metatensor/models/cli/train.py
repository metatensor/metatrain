import argparse
import importlib
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigKeyError

from ..utils.architectures import check_architecture_name, get_default_hypers
from ..utils.data import (
    Dataset,
    DatasetInfo,
    TargetInfo,
    get_atomic_types,
    read_systems,
    read_targets,
)
from ..utils.data.dataset import _train_test_random_split
from ..utils.devices import pick_devices
from ..utils.errors import ArchitectureError
from ..utils.io import check_suffix
from ..utils.omegaconf import (
    BASE_OPTIONS,
    check_options_list,
    check_units,
    expand_dataset_config,
)
from .eval import _eval_targets
from .formatter import CustomHelpFormatter


logger = logging.getLogger(__name__)


def _add_train_model_parser(subparser: argparse._SubParsersAction) -> None:
    """Add `train_model` paramaters to an argparse (sub)-parser."""

    if train_model.__doc__ is not None:
        description = train_model.__doc__.split(r":param")[0]
    else:
        description = None

    # If you change the synopsis of these commands or add new ones adjust the completion
    # script at `src/metatensor/models/share/metatensor-models-completion.bash`.
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
        "-r",
        "--override",
        dest="override_options",
        type=lambda string: OmegaConf.from_dotlist(string.split()),
        help="Command line override flags.",
    )


def train_model(
    options: Union[DictConfig, Dict],
    output: str = "model.pt",
    checkpoint_dir: Union[str, Path] = ".",
    continue_from: Optional[str] = None,
) -> None:
    """Train an atomistic machine learning model using provided ``options``.

    This function sets up the dataset and model architecture, then runs the training
    process. The dataset is prepared by reading structural data and target values from
    specified paths. The model architecture is dynamically imported and instantiated
    based on the configuration. Training is executed with the specified hyperparameters,
    and the trained model is saved to a designated output path.

    :param options: DictConfig containing the training options
    :param output: Path to save the final model
    :param checkpoint_dir: Path to save checkpoints and other intermediate output files
        like the fully expanded training options for a later restart.
    :param continue_from: File to continue training from.
    """
    ###########################
    # LOAD ARCHITECTURE #######
    ###########################

    try:
        architecture_name = options["architecture"]["name"]
    except ConfigKeyError as exc:
        raise ValueError("Architecture name is not defined!") from exc

    check_architecture_name(architecture_name)
    architecture = importlib.import_module(f"metatensor.models.{architecture_name}")

    Model = architecture.__model__
    Trainer = architecture.__trainer__

    ###########################
    # CREATE OPTIONS ##########
    ###########################

    options = OmegaConf.merge(
        BASE_OPTIONS,
        {"architecture": get_default_hypers(architecture_name)},
        options,
    )
    hypers = OmegaConf.to_container(options["architecture"])

    ###########################
    # PROCESS BASE PARAMETERS #
    ###########################

    # process devices
    devices = pick_devices(
        architecture_devices=Model.__supported_devices__,
        desired_device=options["device"],
    )

    # process base_precision/dtypes
    if options["base_precision"] == 64:
        dtype = torch.float64
    elif options["base_precision"] == 32:
        dtype = torch.float32
    elif options["base_precision"] == 16:
        dtype = torch.float16
    else:
        raise ValueError("Only 64, 32 or 16 are possible values for `base_precision`.")

    if dtype not in Model.__supported_dtypes__:
        raise ValueError(
            f"Requested dtype {dtype} is not supported. {architecture_name} only "
            f"supports {Model.__supported_dtypes__}."
        )

    # process random seeds
    if options["seed"] < 0:
        raise ValueError("`seed` should be a positive number")
    else:
        logger.info(f"Random seed of this run is {options['seed']}")
        torch.manual_seed(options["seed"])
        np.random.seed(options["seed"])
        random.seed(options["seed"])
        os.environ["PYTHONHASHSEED"] = str(options["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(options["seed"])
            torch.cuda.manual_seed_all(options["seed"])

    ###########################
    # SETUP TRAINING SET ######
    ###########################

    logger.info("Setting up training set")
    train_options_list = expand_dataset_config(options["training_set"])
    check_options_list(train_options_list)

    train_datasets = []
    for train_options in train_options_list:
        train_systems = read_systems(
            filename=train_options["systems"]["read_from"],
            fileformat=train_options["systems"]["file_format"],
            dtype=dtype,
        )
        train_targets = read_targets(conf=train_options["targets"], dtype=dtype)
        train_datasets.append(Dataset({"system": train_systems, **train_targets}))

    train_size = 1.0

    ###########################
    # SETUP TEST SET ##########
    ###########################

    logger.info("Setting up test set")
    test_options = options["test_set"]
    test_datasets = []
    if isinstance(test_options, float):
        test_size = test_options
        train_size -= test_size

        if test_size < 0 or test_size >= 1:
            raise ValueError(
                "Test set split must be greater or equal than 0 and lesser than 1."
            )

        generator = torch.Generator()
        if options["seed"] is not None:
            generator.manual_seed(options["seed"])

        for i_dataset, train_dataset in enumerate(train_datasets):
            train_dataset_new, test_dataset = _train_test_random_split(
                train_dataset=train_dataset,
                train_size=train_size,
                test_size=test_size,
                generator=generator,
            )

            train_datasets[i_dataset] = train_dataset_new
            test_datasets.append(test_dataset)
    else:
        test_options_list = expand_dataset_config(test_options)
        check_options_list(test_options_list)

        if len(test_options_list) != len(train_options_list):
            raise ValueError(
                f"Test dataset with length {len(test_options_list)} has a different "
                f"size than the train datatset with length {len(train_options_list)}."
            )

        check_units(
            actual_options=test_options_list, desired_options=train_options_list
        )

        for test_options in test_options_list:
            test_systems = read_systems(
                filename=test_options["systems"]["read_from"],
                fileformat=test_options["systems"]["file_format"],
                dtype=dtype,
            )
            test_targets = read_targets(conf=test_options["targets"], dtype=dtype)
            test_dataset = Dataset({"system": test_systems, **test_targets})
            test_datasets.append(test_dataset)

    ###########################
    # SETUP VALIDATION SET ####
    ###########################

    logger.info("Setting up validation set")
    validation_options = options["validation_set"]
    validation_datasets = []
    if isinstance(validation_options, float):
        validation_size = validation_options
        train_size -= validation_size

        if validation_size <= 0 or validation_size >= 1:
            raise ValueError(
                "Validation set split must be greater than 0 and lesser than 1."
            )

        generator = torch.Generator()
        if options["seed"] is not None:
            generator.manual_seed(options["seed"])

        for i_dataset, train_dataset in enumerate(train_datasets):
            train_dataset_new, validation_dataset = _train_test_random_split(
                train_dataset=train_dataset,
                train_size=train_size,
                test_size=validation_size,
                generator=generator,
            )

            train_datasets[i_dataset] = train_dataset_new
            validation_datasets.append(validation_dataset)
    else:
        validation_options_list = expand_dataset_config(validation_options)
        check_options_list(validation_options_list)

        if len(validation_options_list) != len(train_options_list):
            raise ValueError(
                f"Validation dataset with length {len(validation_options_list)} has "
                "a different size than the train datatset with length "
                f"{len(train_options_list)}."
            )

        check_units(
            actual_options=validation_options_list, desired_options=train_options_list
        )

        for validation_options in validation_options_list:
            validation_systems = read_systems(
                filename=validation_options["systems"]["read_from"],
                fileformat=validation_options["systems"]["file_format"],
                dtype=dtype,
            )
            validation_targets = read_targets(
                conf=validation_options["targets"], dtype=dtype
            )
            validation_dataset = Dataset(
                {"system": validation_systems, **validation_targets}
            )
            validation_datasets.append(validation_dataset)

    ###########################
    # SAVE EXPANDED OPTIONS ###
    ###########################

    OmegaConf.save(
        config=options, f=Path(checkpoint_dir) / "options_restart.yaml", resolve=True
    )

    ###########################
    # CREATING DATASET_INFO ###
    ###########################

    # TODO: A more direct way to look up the gradients would be to get them from the
    # configuration dict of the training run.
    gradients: Dict[str, List[str]] = {}
    for train_options in train_options_list:
        for key in train_options["targets"].keys():
            # look inside training sets and find gradients
            for train_dataset in train_datasets:
                if key in train_dataset[0].keys():
                    gradients[key] = train_dataset[0][key].block().gradients_list()

    dataset_info = DatasetInfo(
        length_unit=(
            train_options_list[0]["systems"]["length_unit"]
            if train_options_list[0]["systems"]["length_unit"] is not None
            else ""
        ),  # these units are guaranteed to be the same across all datasets
        atomic_types=get_atomic_types(train_datasets + validation_datasets),
        targets={
            key: TargetInfo(
                quantity=value["quantity"],
                unit=(value["unit"] if value["unit"] is not None else ""),
                per_atom=False,  # TODO: read this from the config
                gradients=gradients[key],
            )
            for train_options in train_options_list
            for key, value in train_options["targets"].items()
        },
    )

    ###########################
    # SETTING UP MODEL ########
    ###########################

    try:
        if continue_from is not None:
            logger.info(f"Loading checkpoint from `{continue_from}`")
            model = Model.load_checkpoint(continue_from)
            model = model.restart(dataset_info)
        else:
            logger.info("Setting up model")
            model = Model(hypers["model"], dataset_info)
    except Exception as e:
        raise ArchitectureError(e)

    ###########################
    # TRAIN MODEL #############
    ###########################

    logger.info("Start training")
    try:
        trainer = Trainer(hypers["training"])
        trainer.train(
            model=model,
            devices=devices,
            train_datasets=train_datasets,
            validation_datasets=validation_datasets,
            checkpoints_dir=str(checkpoint_dir),
        )
    except Exception as e:
        raise ArchitectureError(e)

    ###########################
    # SAVE FINAL MODEL ########
    ###########################

    logger.info("Training finished; save final checkpoint and model")
    output_checked = check_suffix(filename=output, suffix=".pt")
    try:
        model.save_checkpoint(f"{Path(output_checked).stem}.ckpt")
    except Exception as e:
        raise ArchitectureError(e)

    mts_atomistic_model = model.export()
    mts_atomistic_model.export(str(output_checked))

    ###########################
    # EVALUATE FINAL MODEL ####
    ###########################

    for i, train_dataset in enumerate(train_datasets):
        if len(train_datasets) == 1:
            extra_log_message = ""
        else:
            extra_log_message = f" with index {i}"

        logger.info(f"Evaluating training dataset{extra_log_message}")
        _eval_targets(
            mts_atomistic_model,
            train_dataset,
            dataset_info.targets,
            return_predictions=False,
        )

    for i, validation_dataset in enumerate(validation_datasets):
        if len(validation_datasets) == 1:
            extra_log_message = ""
        else:
            extra_log_message = f" with index {i}"

        logger.info(f"Evaluating validation dataset{extra_log_message}")
        _eval_targets(
            mts_atomistic_model,
            validation_dataset,
            dataset_info.targets,
            return_predictions=False,
        )

    for i, test_dataset in enumerate(test_datasets):
        if len(test_datasets) == 1:
            extra_log_message = ""
        else:
            extra_log_message = f" with index {i}"

        logger.info(f"Evaluating test dataset{extra_log_message}")
        _eval_targets(
            mts_atomistic_model,
            test_dataset,
            dataset_info.targets,
            return_predictions=False,
        )
