import argparse
import difflib
import importlib
import logging
import os
import random
import sys
import tempfile
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import List, Optional

import hydra
import numpy as np
import torch
from metatensor.learn.data import Dataset
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigKeyError

from .. import CONFIG_PATH
from ..utils.data import DatasetInfo, TargetInfo, read_systems, read_targets
from ..utils.data.dataset import _train_test_random_split
from ..utils.devices import pick_devices
from ..utils.errors import ArchitectureError
from ..utils.io import export, save
from ..utils.omegaconf import check_options_list, check_units, expand_dataset_config
from .eval import _eval_targets
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
        "-y",
        "--hydra",
        dest="hydra_parameters",
        nargs="+",
        type=str,
        help="Hydra's command line and override flags.",
    )


def check_architecture_name(name: str) -> None:
    """Check if the requested architecture is avalible.

    If the architecture is not found an :func:`ValueError` is raised. If an architecture
    with the same name as an experimental or deprecated architecture exist, this
    architecture is suggested. If no architecture exist the closest architecture is
    given to help debugging typos.

    :param name: name of the architecture
    :raises ValueError: if the architecture is not found
    """
    try:
        if find_spec(f"metatensor.models.{name}") is not None:
            return
        elif find_spec(f"metatensor.models.experimental.{name}") is not None:
            msg = (
                f"Architecture {name!r} is not a stable architecture. An "
                "experimental architecture with the same name was found. Set "
                f"`name: experimental.{name}` in your options file to use this "
                "experimental architecture."
            )
        elif find_spec(f"metatensor.models.deprecated.{name}") is not None:
            msg = (
                f"Architecture {name!r} is not a stable architecture. A "
                "deprecated architecture with the same name was found. Set "
                f"`name: deprecated.{name}` in your options file to use this "
                "deprecated architecture."
            )
    except ModuleNotFoundError:
        arch_avail = [
            f.stem
            for f in (Path(CONFIG_PATH) / "architecture").iterdir()
            if f.is_file()
        ]
        closest_match = difflib.get_close_matches(name, arch_avail, cutoff=0.3)
        msg = (
            f"Architecture {name!r} is not a valid architecture. Do you mean "
            f"{', '.join(closest_match)}?"
        )

    raise ValueError(msg)


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

    check_architecture_name(architecture_name)

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

        if not output.endswith(".pt"):
            warnings.warn(
                "The output file should have a '.pt' extension. The user requested "
                f"the model to be saved as '{output}', but it will be saved as "
                f"'{output}.pt'.",
                stacklevel=1,
            )
            output = f"{output}.pt"

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

    architecture_name = options["architecture"]["name"]
    architecture = importlib.import_module(f"metatensor.models.{architecture_name}")
    architecture_capabilities = architecture.__ARCHITECTURE_CAPABILITIES__

    ###########################
    # PROCESS BASE PARAMETERS #
    ###########################
    devices = pick_devices(
        architecture_devices=architecture_capabilities["supported_devices"],
        desired_device=options["device"],
    )

    # process dtypes
    if options["base_precision"] == 64:
        dtype = torch.float64
    elif options["base_precision"] == 32:
        dtype = torch.float32
    elif options["base_precision"] == 16:
        dtype = torch.float16
    else:
        raise ValueError("Only 64, 32 or 16 are possible values for `base_precision`.")

    if dtype not in architecture_capabilities["supported_dtypes"]:
        raise ValueError(
            f"Requested dtype {dtype} is not supported. {architecture_name} only "
            f"supports {architecture_capabilities['supported_dtypes']}."
        )

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

    ###########################
    # SETUP DATA SETS #########
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
        train_datasets.append(Dataset(system=train_systems, **train_targets))

    train_size = 1.0

    logger.info("Setting up test set")
    test_options = options["test_set"]
    test_datasets = []
    if isinstance(test_options, float):
        test_size = test_options
        train_size -= test_size

        if test_size < 0 or test_size >= 1:
            raise ValueError(
                "Test set split must be greater "
                "than (or equal to) 0 and lesser than 1."
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
            test_dataset = Dataset(system=test_systems, **test_targets)
            test_datasets.append(test_dataset)

    logger.info("Setting up validation set")
    validation_options = options["validation_set"]
    validation_datasets = []
    if isinstance(validation_options, float):
        validation_size = validation_options
        train_size -= validation_size

        if validation_size <= 0 or validation_size >= 1:
            raise ValueError(
                "Validation set split must be greater " "than 0 and lesser than 1."
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
                system=validation_systems, **validation_targets
            )
            validation_datasets.append(validation_dataset)

    # Save fully expanded config
    OmegaConf.save(config=options, f=Path(output_dir) / "options.yaml")

    ###########################
    # SETUP MODEL #############
    ###########################
    logger.info("Setting up model")

    dataset_info = DatasetInfo(
        length_unit=(
            train_options_list[0]["systems"]["length_unit"]
            if train_options_list[0]["systems"]["length_unit"] is not None
            else ""
        ),  # these units are guaranteed to be the same across all datasets
        targets={
            key: TargetInfo(
                quantity=value["quantity"],
                unit=(value["unit"] if value["unit"] is not None else ""),
                per_atom=False,  # TODO: read this from the config
            )
            for train_options in train_options_list
            for key, value in train_options["targets"].items()
        },
    )

    logger.info("Calling architecture trainer")
    try:
        model = architecture.train(
            train_datasets=train_datasets,
            validation_datasets=validation_datasets,
            dataset_info=dataset_info,
            devices=devices,
            hypers=OmegaConf.to_container(options["architecture"]),
            continue_from=options["continue_from"],
            output_dir=output_dir,
        )
    except Exception as e:
        raise ArchitectureError(e)

    save(model, f"{Path(options['output_path']).stem}.ckpt")
    export(model, options["output_path"])
    exported_model = torch.jit.load(options["output_path"])

    for i, train_dataset in enumerate(train_datasets):
        if len(train_datasets) == 1:
            extra_log_message = ""
        else:
            extra_log_message = f" with index {i}"

        logger.info(f"Evaluating training dataset{extra_log_message}")
        eval_options = {
            target: tensormap.block().gradients_list()
            for target, tensormap in train_dataset[0]._asdict().items()
            if target != "system"
        }
        _eval_targets(
            exported_model, train_dataset, eval_options, return_predictions=False
        )

    for i, validation_dataset in enumerate(validation_datasets):
        if len(validation_datasets) == 1:
            extra_log_message = ""
        else:
            extra_log_message = f" with index {i}"

        logger.info(f"Evaluating validation dataset{extra_log_message}")
        eval_options = {
            target: tensormap.block().gradients_list()
            for target, tensormap in validation_dataset[0]._asdict().items()
            if target != "system"
        }
        _eval_targets(
            exported_model, validation_dataset, eval_options, return_predictions=False
        )

    for i, test_dataset in enumerate(test_datasets):
        if len(test_datasets) == 1:
            extra_log_message = ""
        else:
            extra_log_message = f" with index {i}"

        logger.info(f"Evaluating test dataset{extra_log_message}")
        if len(test_dataset) == 0:
            eval_options = {}
        else:
            eval_options = {
                target: tensormap.block().gradients_list()
                for target, tensormap in test_dataset[0]._asdict().items()
                if target != "system"
            }
        _eval_targets(
            exported_model, test_dataset, eval_options, return_predictions=False
        )
