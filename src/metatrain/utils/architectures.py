import difflib
import importlib
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Union

from omegaconf import OmegaConf

from .. import PACKAGE_ROOT
from .hypers import get_hypers_cls, init_with_defaults
from .pydantic import validate_architecture_options


def check_architecture_name(name: str) -> None:
    """Check if the requested architecture is available.

    If the architecture is not found an :func:`ValueError` is raised. If an architecture
    with the same name as an experimental or deprecated architecture exist, this
    architecture is suggested. If no architecture exist the closest architecture is
    given to help debugging typos.

    :param name: name of the architecture
    :raises ValueError: if the architecture is not found
    """
    try:
        if name == "llpr":
            return
        if find_spec(f"metatrain.{name}") is not None:
            return
        elif find_spec(f"metatrain.experimental.{name}") is not None:
            msg = (
                f"Architecture {name!r} is not a stable architecture. An "
                "experimental architecture with the same name was found. Set "
                f"`name: experimental.{name}` in your options file to use this "
                "experimental architecture."
            )
        elif find_spec(f"metatrain.deprecated.{name}") is not None:
            msg = (
                f"Architecture {name!r} is not a stable architecture. A "
                "deprecated architecture with the same name was found. Set "
                f"`name: deprecated.{name}` in your options file to use this "
                "deprecated architecture."
            )
        else:  # not found anywhere, just raise the following except block
            raise ModuleNotFoundError
    except ModuleNotFoundError:
        msg = f"Architecture {name!r} is not a valid architecture."

        closest_match = difflib.get_close_matches(
            word=name, possibilities=find_all_architectures()
        )
        if closest_match:
            msg += f" Did you mean '{closest_match[0]}'?"

    raise ValueError(msg)


def check_architecture_options(
    name: str,
    options: Dict,
) -> None:
    """Verifies that an options instance only contains valid keys

    If the architecture developer does not provide a validation scheme the ``options``
    will not checked.

    :param name: name of the architecture
    :param options: architecture options to check
    """
    architecture = import_architecture(name)

    Model = architecture.__model__
    Trainer = architecture.__trainer__

    validate_architecture_options(
        options,
        Model,
        Trainer,
    )


def get_architecture_name(path: Union[str, Path]) -> str:
    """Name of an architecture based on path to pointing inside an architecture.

    The function should be used to determine the ``ARCHITECTURE_NAME`` based on the name
    of the folder.

    :param path: absolute path of the architecture directory
    :return: architecture name
    :raises ValueError: if ``path`` does not point to a valid architecture directory.

    .. seealso::
        :py:func:`get_architecture_path` to get the relative path within the metatrain
        project of an architecture name.
    """
    path = Path(path)

    if path.is_dir():
        directory = path
    elif path.is_file():
        directory = path.parent
    else:
        raise ValueError(f"`path` {str(path)!r} does not exist")

    architecture_path = directory.relative_to(PACKAGE_ROOT)
    name = str(architecture_path).replace("/", ".")

    try:
        check_architecture_name(name)
    except ValueError as err:
        raise ValueError(
            f"`path` {str(path)!r} does not point to a valid architecture folder"
        ) from err

    return name


def import_architecture(name: str) -> ModuleType:
    """Import an architecture.

    :param name: name of the architecture
    :raises ImportError: if the architecture dependencies are not met
    :return: Imported architecture module
    """
    check_architecture_name(name)
    try:
        module = importlib.import_module(f"metatrain.{name}")
    except ModuleNotFoundError as err:
        # consistent name with pyproject.toml's `optional-dependencies` section
        name_for_deps = name
        if "experimental." in name or "deprecated." in name:
            name_for_deps = ".".join(name.split(".")[1:])

        name_for_deps = name_for_deps.replace("_", "-")

        if err.name and not err.name.startswith(f"metatrain.{name}"):
            raise ModuleNotFoundError(
                f"Trying to import '{name}' but architecture dependencies "
                f"seem not be installed. \n"
                f"Try to install them with `pip install metatrain[{name_for_deps}]`"
            ) from err
        else:
            raise err

    # Import documentation module and set the hypers class for model and trainer
    try:
        documentation = importlib.import_module(f"metatrain.{name}.documentation")
    except ModuleNotFoundError as err:
        if err.name == f"metatrain.{name}.documentation":
            raise ModuleNotFoundError(
                f"Documentation module for architecture '{name}' not found. "
                "Make sure the architecture has a documentation.py file."
            ) from err

    for cls in ["ModelHypers", "TrainerHypers"]:
        if not hasattr(documentation, cls):
            raise ImportError(
                f"Documentation module for architecture '{name}' does not "
                f"contain a '{cls}' class."
            )

    module.__model__.__hypers_cls__ = documentation.ModelHypers
    module.__trainer__.__hypers_cls__ = documentation.TrainerHypers

    return module


def get_architecture_path(name: str) -> Path:
    """Return the relative path to the architecture directory.

    Path based on the ``name`` within the metatrain project directory.

    :param name: name of the architecture
    :return: path to the architecture directory

    .. seealso::
        :py:func:`get_architecture_name` to get the name based on an absolute path of an
        architecture.
    """
    check_architecture_name(name)
    return PACKAGE_ROOT / Path(name.replace(".", "/"))


def find_all_architectures() -> List[str]:
    """Find all currently available architectures.

    To find the architectures the function searches for directories
    that are not part of the shared code of metatrain.

    :return: List of architectures names
    """

    exclude_dirs = ["cli", "experimental", "deprecated", "utils", "share"]

    all_architectures = []

    # Find stable architectures
    for directory in PACKAGE_ROOT.iterdir():
        if (
            (not directory.name.startswith("_"))
            and directory.name not in exclude_dirs
            and (directory / "__init__.py").exists()
        ):
            all_architectures.append(get_architecture_name(directory))

    # Also include experimental and deprecated architectures
    for special_dir in ["experimental", "deprecated"]:
        special_path = PACKAGE_ROOT / special_dir
        for directory in special_path.iterdir():
            if (not directory.name.startswith("_")) and (
                directory / "__init__.py"
            ).exists():
                all_architectures.append(get_architecture_name(directory))

    return all_architectures


def get_default_hypers(name: str) -> Dict:
    """
    Dictionary of the default architecture hyperparameters.

    :param name: Name of the architecture
    :return: Default hyperparameters of the architectures
    """
    check_architecture_name(name)
    architecture = import_architecture(name)

    model_hypers = get_hypers_cls(architecture.__model__)
    trainer_hypers = get_hypers_cls(architecture.__trainer__)

    return {
        "name": name,
        "model": init_with_defaults(model_hypers),
        "training": init_with_defaults(trainer_hypers),
    }


def write_hypers_yaml(name: str, output_path: Path | str) -> None:
    """Write YAML file with defaults for a given architecture.

    Given a model name, this function imports the corresponding
    module, finds out what the hyperparameters are for the model
    and its trainer, and generates a YAML file with the default
    hyperparameters.

    :param name: The model to generate the files for.
    :param output_path: The path to write the YAML file to.
    """
    # Create the dictionary with all default hyperparameters
    yaml_defaults = {"architecture": get_default_hypers(name)}

    conf = OmegaConf.create(yaml_defaults)
    # And write them to a YAML file
    with open(output_path, "w") as f:
        OmegaConf.save(config=conf, f=f)
