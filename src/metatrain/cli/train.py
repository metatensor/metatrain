import argparse
import itertools
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
from metatensor.torch.atomistic import load_atomistic_model
from omegaconf import DictConfig, OmegaConf

from .. import PACKAGE_ROOT
from ..utils.architectures import (
    check_architecture_options,
    get_default_hypers,
    import_architecture,
)
from ..utils.data import (
    DatasetInfo,
    TargetInfoDict,
    get_atomic_types,
    get_dataset,
    get_stats,
)
from ..utils.data.dataset import _train_test_random_split
from ..utils.devices import pick_devices
from ..utils.distributed.logging import is_main_process
from ..utils.errors import ArchitectureError
from ..utils.io import check_file_extension
from ..utils.jsonschema import validate
from ..utils.omegaconf import BASE_OPTIONS, check_units, expand_dataset_config
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
    # script at `src/metatrain/share/metatrain-completion.bash`.
    parser = subparser.add_parser(
        "train",
        description=description,
        formatter_class=CustomHelpFormatter,
    )
    parser.set_defaults(callable="train_model")

    parser.add_argument(
        "options",
        type=str,
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


def _prepare_train_model_args(args: argparse.Namespace) -> None:
    """Prepare arguments for train_model."""
    args.options = OmegaConf.load(args.options)
    # merge/override file options with command line options
    override_options = args.__dict__.pop("override_options")
    if override_options is None:
        override_options = {}

    args.options = OmegaConf.merge(args.options, override_options)


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
    # VALIDATE BASE OPTIONS ###
    ###########################

    # Training, test and validation set options are verified within the
    # `expand_dataset_config()` function.

    with open(PACKAGE_ROOT / "share/schema-base.json", "r") as f:
        schema_base = json.load(f)

    validate(instance=OmegaConf.to_container(options), schema=schema_base)

    ###########################
    # LOAD ARCHITECTURE #######
    ###########################

    architecture_name = options["architecture"]["name"]
    check_architecture_options(
        name=architecture_name, options=OmegaConf.to_container(options["architecture"])
    )
    architecture = import_architecture(architecture_name)

    logger.info(f"Running training for {architecture_name!r} architecture")

    Model = architecture.__model__
    Trainer = architecture.__trainer__

    ###########################
    # MERGE OPTIONS ###########
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
    dtype = getattr(torch, f"float{options['base_precision']}")

    if dtype not in Model.__supported_dtypes__:
        raise ValueError(
            f"Requested dtype {dtype} is not supported. {architecture_name} only "
            f"supports {Model.__supported_dtypes__}."
        )

    # process random seeds
    logger.info(f"Random seed of this run is {options['seed']}")
    torch.manual_seed(options["seed"])
    np.random.seed(options["seed"])
    random.seed(options["seed"])
    os.environ["PYTHONHASHSEED"] = str(options["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(options["seed"])
        torch.cuda.manual_seed_all(options["seed"])

    ############################
    # SET UP TRAINING SET ######
    ############################

    logger.info("Setting up training set")
    options["training_set"] = expand_dataset_config(options["training_set"])

    train_datasets = []
    target_infos = TargetInfoDict()
    for train_options in options["training_set"]:
        dataset, target_info_dict = get_dataset(train_options)
        train_datasets.append(dataset)
        target_infos.update(target_info_dict)

    train_size = 1.0

    ############################
    # SET UP VALIDATION SET ####
    ############################

    logger.info("Setting up validation set")
    val_datasets = []
    if isinstance(options["validation_set"], float):
        val_size = options["validation_set"]
        train_size -= val_size

        for i_dataset, train_dataset in enumerate(train_datasets):
            train_dataset_new, val_dataset = _train_test_random_split(
                train_dataset=train_dataset,
                train_size=train_size,
                test_size=val_size,
            )

            train_datasets[i_dataset] = train_dataset_new
            val_datasets.append(val_dataset)
    else:
        options["validation_set"] = expand_dataset_config(options["validation_set"])

        if len(options["validation_set"]) != len(options["training_set"]):
            raise ValueError(
                f"Validation dataset with length {len(options['validation_set'])} has "
                "a different size than the training datatset with length "
                f"{len(options['training_set'])}."
            )

        check_units(
            actual_options=options["validation_set"],
            desired_options=options["training_set"],
        )

        for valid_options in options["validation_set"]:
            dataset, _ = get_dataset(valid_options)
            val_datasets.append(dataset)

    ############################
    # SET UP TEST SET ##########
    ############################

    logger.info("Setting up test set")
    test_datasets = []
    if isinstance(options["test_set"], float):
        test_size = options["test_set"]
        train_size -= test_size

        for i_dataset, train_dataset in enumerate(train_datasets):
            train_dataset_new, test_dataset = _train_test_random_split(
                train_dataset=train_dataset,
                train_size=train_size,
                test_size=test_size,
            )

            train_datasets[i_dataset] = train_dataset_new
            test_datasets.append(test_dataset)
    else:
        options["test_set"] = expand_dataset_config(options["test_set"])

        if len(options["test_set"]) != len(options["training_set"]):
            raise ValueError(
                f"Test dataset with length {len(options['test_set'])} has a different "
                f"size than the training datatset with length "
                f"{len(options['training_set'])}."
            )

        check_units(
            actual_options=options["test_set"],
            desired_options=options["training_set"],
        )

        for test_options in options["test_set"]:
            dataset, _ = get_dataset(test_options)
            test_datasets.append(dataset)

    ###########################
    # CREATE DATASET_INFO #####
    ###########################

    atomic_types = get_atomic_types(train_datasets + val_datasets)

    dataset_info = DatasetInfo(
        length_unit=options["training_set"][0]["systems"]["length_unit"],
        atomic_types=atomic_types,
        targets=target_infos,
    )

    ###########################
    # PRINT DATASET STATS #####
    ###########################

    for i, train_dataset in enumerate(train_datasets):
        if len(train_datasets) == 1:
            index = ""
        else:
            index = f" {i}"
        logger.info(
            f"Training dataset{index}:\n    {get_stats(train_dataset, dataset_info)}"
        )

    for i, val_dataset in enumerate(val_datasets):
        if len(val_datasets) == 1:
            index = ""
        else:
            index = f" {i}"
        logger.info(
            f"Validation dataset{index}:\n    {get_stats(val_dataset, dataset_info)}"
        )

    for i, test_dataset in enumerate(test_datasets):
        if len(test_datasets) == 1:
            index = ""
        else:
            index = f" {i}"
        logger.info(
            f"Test dataset{index}:\n    {get_stats(test_dataset, dataset_info)}"
        )

    ###########################
    # SAVE EXPANDED OPTIONS ###
    ###########################

    OmegaConf.save(
        config=options, f=Path(checkpoint_dir) / "options_restart.yaml", resolve=True
    )

    ###########################
    # SETTING UP MODEL ########
    ###########################

    logger.info("Setting up model")
    try:
        if continue_from is not None:
            logger.info(f"Loading checkpoint from `{continue_from}`")
            trainer = Trainer.load_checkpoint(continue_from, hypers["training"])
            model = Model.load_checkpoint(continue_from)
            model = model.restart(dataset_info)
        else:
            model = Model(hypers["model"], dataset_info)
            trainer = Trainer(hypers["training"])
    except Exception as e:
        raise ArchitectureError(e)

    ###########################
    # TRAIN MODEL #############
    ###########################

    logger.info("Calling trainer")
    try:
        trainer.train(
            model=model,
            dtype=dtype,
            devices=devices,
            train_datasets=train_datasets,
            val_datasets=val_datasets,
            checkpoint_dir=str(checkpoint_dir),
        )
    except Exception as e:
        raise ArchitectureError(e)

    if not is_main_process():
        return  # only save and evaluate on the main process

    ###########################
    # SAVE FINAL MODEL ########
    ###########################

    output_checked = check_file_extension(filename=output, extension=".pt")
    logger.info(
        "Training finished, saving final checkpoint "
        f"to `{str(Path(output_checked).stem)}.ckpt`"
    )
    try:
        trainer.save_checkpoint(model, f"{Path(output_checked).stem}.ckpt")
    except Exception as e:
        raise ArchitectureError(e)

    mts_atomistic_model = model.export()
    extensions_path = "extensions/"

    logger.info(
        f"Exporting model to `{output_checked}` and extensions to `{extensions_path}`"
    )
    # get device from the model. This device could be different from devices[0]
    # defined above in the case of multi-GPU and/or distributed training
    final_device = next(
        itertools.chain(
            mts_atomistic_model.parameters(),
            mts_atomistic_model.buffers(),
        )
    ).device
    mts_atomistic_model.save(str(output_checked), collect_extensions=extensions_path)
    # the model is first saved and then reloaded 1) for good practice and 2) because
    # MetatensorAtomisticModel only torchscripts (makes faster) during save()

    ###########################
    # EVALUATE FINAL MODEL ####
    ###########################

    mts_atomistic_model = load_atomistic_model(
        str(output_checked), extensions_directory=extensions_path
    )
    mts_atomistic_model = mts_atomistic_model.to(final_device)

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

    for i, val_dataset in enumerate(val_datasets):
        if len(val_datasets) == 1:
            extra_log_message = ""
        else:
            extra_log_message = f" with index {i}"

        logger.info(f"Evaluating validation dataset{extra_log_message}")
        _eval_targets(
            mts_atomistic_model,
            val_dataset,
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
