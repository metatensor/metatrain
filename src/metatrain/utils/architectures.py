import difflib
import importlib
import sys
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Union

import yaml

from .. import PACKAGE_ROOT
from .hypers import init_with_defaults
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
    hypers_classes = get_hypers_classes(name)
    validate_architecture_options(
        options, hypers_classes["model"], hypers_classes["trainer"]
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
    name = ".".join(architecture_path.parts)

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
        return importlib.import_module(f"metatrain.{name}")
    except ImportError as err:
        # consistent name with pyproject.toml's `optional-dependencies` section
        name_for_deps = name
        if "experimental." in name or "deprecated." in name:
            name_for_deps = ".".join(name.split(".")[1:])

        name_for_deps = name_for_deps.replace("_", "-")

        if (
            isinstance(err, ModuleNotFoundError)
            and err.name
            and not err.name.startswith(f"metatrain.{name}")
        ):
            raise ModuleNotFoundError(
                f"Trying to import '{name}' but architecture dependencies "
                f"seem not be installed. \n"
                f"Try to install them with `pip install metatrain[{name_for_deps}]`"
            ) from err
        else:
            raise ImportError(
                f"An error occurred while importing the architecture '{name}'. "
                "This is likely due to a broken installation. Reinstalling metatrain "
                "and its dependencies might help: "
                f"`pip install metatrain[{name_for_deps}]`"
            ) from err


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


def preload_documentation_module(name: str) -> ModuleType:
    """This preloads the documentation module for a given architecture.

    It imports the `documentation.py` file in an isolated manner and
    adds it to `sys.modules`.

    The reason one might do this is because the documentation module
    does not have extra dependencies, so importing it separately is
    always possible, while if we didn't preload it, importing the
    documentation would trigger the architecture's `__init__.py`
    which might have extra dependencies that are not installed.

    Doing this preloading is useful especially in the context of
    generating the documentation, where we want to be able to
    document architectures even if their dependencies are not
    installed.

    :param name: Name of the architecture
    :return: The documentation module for the architecture.
    """
    file_path = get_architecture_path(name) / "documentation.py"
    if not file_path.exists():
        raise FileNotFoundError(
            f"The documentation.py file for architecture '{name}' was not found. "
            "Cannot load the architecture's hyperparameter specification."
        )
    spec = importlib.util.spec_from_file_location(
        f"metatrain.{name}.documentation", file_path
    )
    assert spec is not None  # for mypy
    documentation = importlib.util.module_from_spec(spec)
    assert spec.loader is not None  # for mypy
    spec.loader.exec_module(documentation)
    sys.modules[f"metatrain.{name}.documentation"] = documentation
    return documentation


def get_hypers_classes(name: str) -> Dict[str, type]:
    """
    Returns the default architecture hyperparameters.

    :param name: Name of the architecture
    :return: Default hyperparameters of the architectures
    """
    check_architecture_name(name)

    try:
        documentation = importlib.import_module(f"metatrain.{name}.documentation")
    except ModuleNotFoundError as err:
        if err.name == f"metatrain.{name}.documentation":
            raise ModuleNotFoundError(
                f"Documentation module for architecture '{name}' not found. "
                "Make sure the architecture has a documentation.py file."
            ) from err

    documentation = importlib.import_module(f"metatrain.{name}.documentation")

    return {
        "model": documentation.ModelHypers,
        "trainer": documentation.TrainerHypers,
    }


def get_default_hypers(name: str) -> Dict:
    """
    Returns the default architecture hyperparameters.

    :param name: Name of the architecture
    :return: Default hyperparameters of the architectures
    """
    check_architecture_name(name)

    hypers_classes = get_hypers_classes(name)

    return {
        "name": name,
        "model": init_with_defaults(hypers_classes["model"]),
        "training": init_with_defaults(hypers_classes["trainer"]),
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
