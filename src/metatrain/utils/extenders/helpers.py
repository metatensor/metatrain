import copy
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

import torch
from omegaconf import OmegaConf

from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.hypers import init_with_defaults

from .abc import ExtenderInterface


def find_all_extenders() -> list[str]:
    """
    Returns a list of all extenders in the metatrain.utils.extenders package.

    :return: A list of extender names.
    """

    extenders_root = Path(__file__).parent

    # Get all the directories in the extenders package
    return [
        p.name
        for p in extenders_root.iterdir()
        if (p.is_dir() and (p / "__init__.py").exists() and not p.name == "testing")
    ]


def load_extender(extender_name: str) -> type[ExtenderInterface]:
    """
    Loads an extender from the metatrain.utils.extenders package.

    :param extender_name: The name of the extender to load.
    :return: The extender class.
    """
    if extender_name not in find_all_extenders():
        raise ValueError(f"Unknown extender: {extender_name}")

    # Import the extender module
    extender_module = importlib.import_module(
        f"metatrain.utils.extenders.{extender_name}"
    )
    return extender_module.__extender__


def preload_documentation_module(name: str) -> ModuleType:
    """This preloads the documentation module for a given extender.

    It imports the `documentation.py` file in an isolated manner and
    adds it to `sys.modules`.

    The reason one might do this is because the documentation module
    does not have extra dependencies, so importing it separately is
    always possible, while if we didn't preload it, importing the
    documentation would trigger the extender's `__init__.py`
    which might have extra dependencies that are not installed.

    Doing this preloading is useful especially in the context of
    generating the documentation, where we want to be able to
    document extenders even if their dependencies are not
    installed.

    :param name: Name of the extender
    :return: The documentation module for the extender.
    """
    file_path = Path(__file__).parent / name / "documentation.py"
    if not file_path.exists():
        raise FileNotFoundError(
            f"The documentation.py file for extender '{name}' was not found. "
            "Cannot load the extender's hyperparameter specification."
        )
    spec = importlib.util.spec_from_file_location(
        f"metatrain.utils.extenders.{name}.documentation", file_path
    )
    assert spec is not None  # for mypy
    documentation = importlib.util.module_from_spec(spec)
    assert spec.loader is not None  # for mypy
    spec.loader.exec_module(documentation)
    sys.modules[f"metatrain.utils.extenders.{name}.documentation"] = documentation
    return documentation


def get_hypers_class(name: str) -> type:
    """
    Returns the hypers classes for a given extender.

    :param name: Name of the extender.
    :return: The hypers class for the extender.
    """
    documentation = preload_documentation_module(name)
    return documentation.Hypers


def get_default_hypers(name: str, base_precision: Optional[int] = None) -> dict:
    """Returns the default extender hyperparameters.

    When *base_precision* is ``None`` (the default), the returned dict may
    contain ``"${base_precision}"`` interpolation strings in precision fields.
    These resolve automatically after ``OmegaConf.merge`` with a config that
    provides ``base_precision`` at the root level.

    When *base_precision* is an integer (16, 32, or 64), all interpolations are
    resolved before returning so callers get a plain dict of concrete values.

    :param name: Name of the extender
    :param base_precision: If given, resolve ``${base_precision}`` interpolations
        to this value before returning.
    :return: Default hyperparameters of the extender
    """

    hypers_class = get_hypers_class(name)

    defaults = init_with_defaults(hypers_class)

    if base_precision is not None:
        cfg = OmegaConf.create({"base_precision": base_precision, **defaults})
        container = OmegaConf.to_container(cfg, resolve=True)
        container.pop("base_precision")
        return container

    return copy.deepcopy(defaults)


def write_hypers_yaml(
    name: str, output_path: Path | str, include_name: bool = False
) -> None:
    """Write YAML file with defaults for a given extender.

    Given an extender name, this function imports the corresponding
    module, finds out what the hyperparameters are for the extender
    and its trainer, and generates a YAML file with the default
    hyperparameters.

    :param name: The extender to generate the files for.
    :param output_path: The path to write the YAML file to.
    :param include_name: If True, the YAML file will contain a top-level
        key with the extender name, mimicking the input that needs to be
        provided to use the extender.
    """
    # Create the dictionary with all default hyperparameters
    yaml_defaults = get_default_hypers(name)

    if include_name:
        yaml_defaults = {name: yaml_defaults}

    conf = OmegaConf.create(yaml_defaults)
    # And write them to a YAML file
    with open(output_path, "w") as f:
        OmegaConf.save(config=conf, f=f)


class UnavailableOutputError(Exception):
    """Should be used when an extender is asked to produce an output
    but the extender is not able to find the layout of such output.

    This is because the function that sets up extenders will go through
    the other extenders and then retry this one. Maybe some other extender
    has requested an input that matches this output's name and then
    the layout will now be available.
    """


def setup_extenders(
    dataset_info: DatasetInfo,
    extenders_hypers: dict[str, dict[str, str] | str],
) -> tuple[
    list[torch.nn.Module],
    dict[str, TargetInfo],
]:
    """
    Setup the post-processing extenders to apply to the outputs of the model.

    :param dataset_info: The dataset information, which contains target
        information and also the extenders specification.
    :param extenders_hypers: The hyperparameters for the extenders.

    :return: A tuple containing the list of post-processing extenders, and a
        dictionary with the outputs that the model should produce before the
        extenders are applied.
    """
    dataset_info = dataset_info.copy()
    model_outputs = dataset_info.targets.copy()

    all_extender_outputs = []
    extenders = []

    def _set_up(extenders_hypers: dict, dataset_info: DatasetInfo) -> tuple[dict, list]:
        unavailable_out_extenders = {}
        unavailable_errors = []
        for extender_name, extender_hypers in extenders_hypers.items():
            extender_class = load_extender(extender_name)

            # Get extender and add it to the list of extenders.
            try:
                extender = extender_class(extender_hypers, dataset_info)
            except UnavailableOutputError as e:
                unavailable_out_extenders[extender_name] = extender_hypers
                unavailable_errors.append(e)
                continue
            extenders.append(extender)

            # The model will not need to produce the outputs that this extender
            # takes care of, so remove them from the list of model outputs.
            extender_outputs = extender.supported_outputs()
            for output_name in extender_outputs:
                if output_name in all_extender_outputs:
                    raise ValueError(
                        f"Output '{output_name}' is already produced by another "
                        "extender. A given output can only be produced by one extender."
                    )
                model_outputs.pop(output_name, None)
            all_extender_outputs.extend(list(extender_outputs.keys()))

            # Add the inputs that the extender requests to the model outputs,
            # unless they are already present somewhere.
            requested = extender.requested_extender_inputs()
            extra_data_info = dataset_info.extra_data
            for req_name, req_info in requested.items():
                if req_name in model_outputs:
                    continue
                elif req_name in all_extender_outputs:
                    continue
                elif req_name in extra_data_info:
                    continue
                else:
                    model_outputs[req_name] = req_info

        return unavailable_out_extenders, unavailable_errors

    extenders_to_set_up = extenders_hypers
    while True:
        new_extenders_to_set_up, errors = _set_up(extenders_to_set_up, dataset_info)
        if len(new_extenders_to_set_up) == 0:
            break
        elif len(new_extenders_to_set_up) == len(extenders_to_set_up):
            raise ValueError(
                "Some extenders could not be set up because they requested outputs "
                "that are not present either in the targets, extra data or "
                "in other extenders' inputs. \nThe errors were the following:\n - "
                + "\n - ".join(str(e) for e in errors)
            )
        # We managed to set up some extenders in this iteration, do another iteration
        # to see if we can set up more extenders now that some outputs are available.
        extenders_to_set_up = new_extenders_to_set_up
        dataset_info.targets.update(model_outputs)

    # Reorder extenders to match the order in which they were specified in the hypers.
    req_extenders_classes = [load_extender(name) for name in extenders_hypers.keys()]
    extenders.sort(key=lambda e: req_extenders_classes.index(type(e)))

    return extenders, model_outputs


def restart_extenders(
    instantiated_extenders: list[ExtenderInterface],
    dataset_info: DatasetInfo,
    new_extenders_hypers: dict[str, dict[str, str] | str],
) -> tuple[list[ExtenderInterface], dict[str, TargetInfo], dict]:
    """
    Restart extenders with new dataset information.

    :param instantiated_extenders: List of existing extenders to restart.
    :param dataset_info: New dataset information to use for restarting the extenders.
    :param new_extenders_hypers: The extender hypers passed for the extenders restart.
    :return: A tuple containing:
      - the list of restarted extenders,
      - a dictionary with the outputs that the model should produce before
        the extenders are applied,
      - the extender hyperparameters that will regenerate the returned extenders
        if passed to setup_extenders(). These should be stored in the checkpoint
        of the model.
    """
    if len(instantiated_extenders) > 0:
        raise ValueError("Restarting from previous extenders is not supported yet.")

    extenders, model_outs = setup_extenders(dataset_info, new_extenders_hypers)

    return extenders, model_outs, new_extenders_hypers
