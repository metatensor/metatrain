import argparse
import itertools
import json
import logging
import os
import random
import re
import shutil
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .. import PACKAGE_ROOT
from ..utils.abc import ModelInterface, TrainerInterface
from ..utils.architectures import (
    check_architecture_options,
    get_default_hypers,
    import_architecture,
)
from ..utils.data import (
    DatasetInfo,
    TargetInfo,
    get_atomic_types,
    get_dataset,
    get_stats,
)
from ..utils.data.dataset import _save_indices, _train_test_random_split
from ..utils.devices import pick_devices
from ..utils.distributed.logging import is_main_process
from ..utils.errors import ArchitectureError
from ..utils.io import (
    check_file_extension,
    load_model,
    model_from_checkpoint,
    trainer_from_checkpoint,
)
from ..utils.jsonschema import validate
from ..utils.logging import ROOT_LOGGER, WandbHandler, human_readable
from ..utils.omegaconf import BASE_OPTIONS, check_units, expand_dataset_config
from .eval import _eval_targets
from .export import _has_extensions
from .formatter import CustomHelpFormatter


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
        help="Options YAML file for training.",
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
        "-e",
        "--extensions",
        dest="extensions",
        type=str,
        required=False,
        default="extensions/",
        help=(
            "Folder where the extensions of the model, if any, will be collected "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--restart",
        dest="restart_from",
        type=_process_restart_from,
        required=False,
        help=(
            "Checkpoint file (.ckpt) to continue interrupted training. Set to `'auto'` "
            "to take the most recent checkpoint from the outputs directory."
        ),
    )
    parser.add_argument(
        "-r",
        "--override",
        dest="override_options",
        action="append",
        help="Command-line override flags.",
        default=[],
    )


def _prepare_train_model_args(args: argparse.Namespace) -> None:
    """Prepare arguments for train_model."""
    args.options = OmegaConf.load(args.options)
    # merge/override file options with command line options
    override_options = args.__dict__.pop("override_options")
    override_options = OmegaConf.from_dotlist(override_options)

    args.options = OmegaConf.merge(args.options, override_options)


def _process_restart_from(restart_from: str) -> Optional[Union[str, Path]]:
    if restart_from != "auto":
        return restart_from

    pattern = re.compile(r".*\d{4}-\d{2}-\d{2}/\d{2}-\d{2}-\d{2}/*")
    checkpoints = sorted(
        (f for f in Path("outputs").glob("*/*/*.ckpt") if pattern.match(str(f))),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    return checkpoints[0] if checkpoints else None


def train_model(
    options: Union[DictConfig, Dict],
    output: Union[str, Path] = "model.pt",
    extensions: Union[str, Path] = "extensions/",
    checkpoint_dir: Union[str, Path] = ".",
    restart_from: Optional[Union[str, Path]] = None,
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
    :param restart_from: File to continue training from.
    """

    output = Path(check_file_extension(filename=output, extension=".pt"))
    extensions = Path(extensions)
    checkpoint_dir = Path(checkpoint_dir)

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

    logging.info(f"Running training for {architecture_name!r} architecture")

    Model = architecture.__model__
    if not issubclass(Model, ModelInterface):
        raise TypeError(
            f"Model class for {architecture_name} must be a subclass of "
            " `metatrain.utils.abc.ModelInterface`"
        )

    Trainer = architecture.__trainer__
    if not issubclass(Trainer, TrainerInterface):
        raise TypeError(
            f"Trainer class for {architecture_name} must be a subclass of "
            " `metatrain.utils.abc.TrainerInterface`"
        )

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
    logging.info(f"Random seed of this run is {options['seed']}")
    torch.manual_seed(options["seed"])
    np.random.seed(options["seed"])
    random.seed(options["seed"])
    os.environ["PYTHONHASHSEED"] = str(options["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(options["seed"])
        torch.cuda.manual_seed_all(options["seed"])

    # setup wandb logging
    if hasattr(options, "wandb"):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "Wandb is enabled but not installed. "
                "Please install wandb using `pip install wandb` to use this logger."
            )
        logging.info("Setting up wandb logging")

        run = wandb.init(
            **options["wandb"], config=OmegaConf.to_container(options, resolve=True)
        )
        ROOT_LOGGER.addHandler(WandbHandler(run))

    ############################
    # SET UP TRAINING SET ######
    ############################

    logging.info("Setting up training set")
    options["training_set"] = expand_dataset_config(options["training_set"])

    train_datasets = []
    target_info_dict: Dict[str, TargetInfo] = {}
    extra_data_info_dict: Dict[str, TargetInfo] = {}
    for train_options in options["training_set"]:  # loop over training sets
        dataset, target_info_dict_single, extra_data_info_dict_single = get_dataset(
            train_options
        )
        train_datasets.append(dataset)

        intersecting_keys = target_info_dict.keys() & target_info_dict_single.keys()
        for key in intersecting_keys:
            if target_info_dict[key] != target_info_dict_single[key]:
                raise ValueError(
                    f"Target information for key {key} differs between training sets. "
                    f"Got {target_info_dict[key]} and {target_info_dict_single[key]}."
                )
        target_info_dict.update(target_info_dict_single)

        intersecting_keys = (
            extra_data_info_dict.keys() & extra_data_info_dict_single.keys()
        )
        for key in intersecting_keys:
            if extra_data_info_dict[key] != extra_data_info_dict_single[key]:
                raise ValueError(
                    f"Extra data information for key {key} differs between training "
                    f"sets. Got {extra_data_info_dict[key]} and"
                    f" {extra_data_info_dict_single[key]}."
                )
        extra_data_info_dict.update(extra_data_info_dict_single)

    train_size = 1.0

    ############################
    # SET UP VALIDATION SET ####
    ############################

    logging.info("Setting up validation set")
    val_datasets = []
    train_indices = []
    val_indices = []
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
            train_indices.append(train_dataset_new.indices)
            val_indices.append(val_dataset.indices)
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
            dataset, _, _ = get_dataset(valid_options)
            val_datasets.append(dataset)
            train_indices.append(None)
            val_indices.append(None)

    ############################
    # SET UP TEST SET ##########
    ############################

    logging.info("Setting up test set")
    test_datasets = []
    test_indices = []
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
            there_was_no_validation_split = train_indices[i_dataset] is None
            new_train_indices = (
                train_dataset_new.indices
                if there_was_no_validation_split
                else [train_indices[i_dataset][i] for i in train_dataset_new.indices]
            )
            test_indices.append(
                test_dataset.indices
                if there_was_no_validation_split
                else [train_indices[i_dataset][i] for i in test_dataset.indices]
            )
            train_indices[i_dataset] = new_train_indices
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
            dataset, _, _ = get_dataset(test_options)
            test_datasets.append(dataset)
            test_indices.append(None)

    ############################################
    # SAVE TRAIN, VALIDATION, TEST INDICES #####
    ############################################

    if is_main_process():
        _save_indices(train_indices, val_indices, test_indices, checkpoint_dir)

    ###########################
    # CREATE DATASET_INFO #####
    ###########################
    if options["architecture"].get("atomic_types") is None:  # infer from datasets
        logging.info("Atomic types inferred from training and validation datasets")
        atomic_types = get_atomic_types(train_datasets + val_datasets)
    else:  # use explicitly defined atomic types
        logging.info("Atomic types explicitly defined in options.yaml")
        atomic_types = sorted(options["architecture"]["atomic_types"])

    logging.info(f"Model defined for atomic types: {atomic_types}")

    dataset_info = DatasetInfo(
        length_unit=options["training_set"][0]["systems"]["length_unit"],
        atomic_types=atomic_types,
        targets=target_info_dict,
        extra_data=extra_data_info_dict,
    )

    ###########################
    # PRINT DATASET STATS #####
    ###########################

    for i, train_dataset in enumerate(train_datasets):
        if len(train_datasets) == 1:
            index = ""
        else:
            index = f" {i}"
        logging.info(
            f"Training dataset{index}:\n    {get_stats(train_dataset, dataset_info)}"
        )

    for i, val_dataset in enumerate(val_datasets):
        if len(val_datasets) == 1:
            index = ""
        else:
            index = f" {i}"
        logging.info(
            f"Validation dataset{index}:\n    {get_stats(val_dataset, dataset_info)}"
        )

    for i, test_dataset in enumerate(test_datasets):
        if len(test_datasets) == 1:
            index = ""
        else:
            index = f" {i}"
        logging.info(
            f"Test dataset{index}:\n    {get_stats(test_dataset, dataset_info)}"
        )

    ###########################
    # SAVE EXPANDED OPTIONS ###
    ###########################

    if is_main_process():
        logging.info(
            "Restart options: "
            f"{checkpoint_dir.absolute().resolve() / 'options_restart.yaml'}"
        )
        OmegaConf.save(
            config=options,
            f=checkpoint_dir / "options_restart.yaml",
            resolve=True,
        )

    ###########################
    # SETTING UP MODEL ########
    ###########################

    logging.info("Setting up model")

    # Resolving the model initialization options
    if restart_from is not None:
        training_context = "restart"
    elif "finetune" in hypers["training"]:
        if "read_from" not in hypers["training"]["finetune"]:
            raise ValueError(
                "Finetuning is enabled but no checkpoint was provided. Please "
                "provide one using the `read_from` option in the `finetune` "
                "section."
            )
        restart_from = hypers["training"]["finetune"]["read_from"]
        training_context = "finetune"
    else:
        training_context = None

    try:
        if training_context == "restart" and restart_from is not None:
            logging.info(f"Restarting training from '{restart_from}'")
            checkpoint = torch.load(
                restart_from, weights_only=False, map_location="cpu"
            )
            try:
                model = model_from_checkpoint(checkpoint, context="restart")
            except Exception as e:
                raise ValueError(
                    f"The file {restart_from} does not contain a valid checkpoint for "
                    f"the '{architecture_name}' architecture"
                ) from e
            model = model.restart(dataset_info)
            try:
                trainer = trainer_from_checkpoint(
                    checkpoint=checkpoint,
                    hypers=hypers["training"],
                    context=training_context,  # type: ignore
                )
            except Exception as e:
                raise ValueError(
                    f"The file {restart_from} does not contain a valid checkpoint for "
                    f"the '{architecture_name}' trainer state"
                ) from e
        elif training_context == "finetune" and restart_from is not None:
            logging.info(f"Starting finetuning from '{restart_from}'")
            checkpoint = torch.load(
                restart_from, weights_only=False, map_location="cpu"
            )
            try:
                model = model_from_checkpoint(checkpoint, context="finetune")
            except Exception as e:
                raise ValueError(
                    f"The file {restart_from} does not contain a valid checkpoint for "
                    f"the '{architecture_name}' architecture"
                ) from e
            model = model.restart(dataset_info)
            trainer = Trainer(hypers["training"])
        else:
            logging.info("Starting training from scratch")
            model = Model(hypers["model"], dataset_info)
            trainer = Trainer(hypers["training"])
    except Exception as e:
        raise ArchitectureError(e) from e

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(
        (
            f"The model has {human_readable(n_params)} parameters "
            f"(actual number: {n_params})."
        )
    )

    ###########################
    # TRAIN MODEL #############
    ###########################

    logging.info("Calling trainer")
    logging.info(
        "Intermediate checkpoints (if available): "
        f"{checkpoint_dir.absolute().resolve()}"
    )
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
        raise ArchitectureError(e) from e

    if not is_main_process():
        return  # only save and evaluate on the main process

    ###########################
    # SAVE FINAL MODEL ########
    ###########################

    logging.info("Training finished!")

    checkpoint_output = output.with_suffix(".ckpt")
    try:
        trainer.save_checkpoint(model, checkpoint_output)
    except Exception as e:
        raise ArchitectureError(e)

    if checkpoint_output.exists():
        # Reload ensuring (best) model intended for inference
        model = load_model(checkpoint_output)

        logging.info(f"Final checkpoint: {checkpoint_output.absolute().resolve()}")

    mts_atomistic_model = model.export()
    # Final device could be different from devices[0] defined above in the case of
    # multi-GPU and/or distributed training
    final_device = next(
        itertools.chain(
            mts_atomistic_model.parameters(),
            mts_atomistic_model.buffers(),
        )
    ).device

    # model is first saved and then reloaded 1) for good practice and 2) because
    # `AtomisticModel` only torchscripts (makes faster) during `save()`
    mts_atomistic_model.save(
        file=output,
        collect_extensions=extensions if _has_extensions() else None,
    )

    logging.info(f"Exported model: {output.absolute().resolve()}")
    if extensions.exists():
        logging.info(f"Extensions path: {extensions.absolute().resolve()}")

    if checkpoint_dir.absolute().resolve() != Path.cwd():
        shutil.copy(output, checkpoint_dir / output)
        if checkpoint_output.exists():
            shutil.copy(checkpoint_output, checkpoint_dir / checkpoint_output)

    ###########################
    # EVALUATE FINAL MODEL ####
    ###########################

    mts_atomistic_model = load_model(
        path=output,
        extensions_directory=extensions if _has_extensions() else None,
    )
    mts_atomistic_model = mts_atomistic_model.to(final_device)

    batch_size = _get_batch_size_from_hypers(hypers)
    if batch_size is None:
        logging.warning(
            "Could not find batch size in hypers dictionary. "
            "Using default value of 1 for final evaluation."
        )
        batch_size = 1
    else:
        logging.info(f"Running final evaluation with batch size {batch_size}")

    for i, train_dataset in enumerate(train_datasets):
        if len(train_datasets) == 1:
            extra_log_message = ""
        else:
            extra_log_message = f" with index {i}"

        logging.info(f"Evaluating training dataset{extra_log_message}")
        _eval_targets(
            mts_atomistic_model,
            train_dataset,
            dataset_info.targets,
            batch_size=batch_size,
        )

    for i, val_dataset in enumerate(val_datasets):
        if len(val_datasets) == 1:
            extra_log_message = ""
        else:
            extra_log_message = f" with index {i}"

        logging.info(f"Evaluating validation dataset{extra_log_message}")
        _eval_targets(
            mts_atomistic_model,
            val_dataset,
            dataset_info.targets,
            batch_size=batch_size,
        )

    for i, test_dataset in enumerate(test_datasets):
        if len(test_datasets) == 1:
            extra_log_message = ""
        else:
            extra_log_message = f" with index {i}"

        logging.info(f"Evaluating test dataset{extra_log_message}")
        _eval_targets(
            mts_atomistic_model,
            test_dataset,
            dataset_info.targets,
            batch_size=batch_size,
        )


def _get_batch_size_from_hypers(hypers: Union[Dict, DictConfig]) -> Optional[int]:
    # Recursively searches for "batch_size", "batch-size", "batch size", "batchsize",
    # or their upper-case equivalents in the hypers dictionary and returns the value.
    # If not found, it returns None.
    for key in hypers.keys():
        value = hypers[key]
        if isinstance(value, dict) or isinstance(value, DictConfig):
            batch_size = _get_batch_size_from_hypers(value)
            if batch_size is not None:
                return batch_size
        if (
            key.lower().replace("_", "").replace("-", "").replace(" ", "")
            == "batchsize"
        ):
            return value
    return None
