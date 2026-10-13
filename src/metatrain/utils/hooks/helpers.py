import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional, cast

import torch
from omegaconf import OmegaConf

from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.hypers import init_with_defaults

from .abc import HookInterface


def find_all_hooks() -> list[str]:
    """
    Returns a list of all hooks in the metatrain.utils.hooks package.

    :return: A list of hook names.
    """

    hooks_root = Path(__file__).parent

    # Get all the directories in the hooks package
    return [
        p.name
        for p in hooks_root.iterdir()
        if (p.is_dir() and (p / "__init__.py").exists() and not p.name == "testing")
    ]


def load_hook(hook_name: str) -> type[HookInterface]:
    """
    Loads a hook from the metatrain.utils.hooks package.

    :param hook_name: The name of the hook to load.
    :return: The hook class.
    """
    if hook_name not in find_all_hooks():
        raise ValueError(f"Unknown hook: {hook_name}")

    # Import the hook module
    hook_module = importlib.import_module(f"metatrain.utils.hooks.{hook_name}")
    return hook_module.__hook__


def preload_documentation_module(name: str) -> ModuleType:
    """This preloads the documentation module for a given hook.

    It imports the `documentation.py` file in an isolated manner and
    adds it to `sys.modules`.

    The reason one might do this is because the documentation module
    does not have extra dependencies, so importing it separately is
    always possible, while if we didn't preload it, importing the
    documentation would trigger the hook's `__init__.py`
    which might have extra dependencies that are not installed.

    Doing this preloading is useful especially in the context of
    generating the documentation, where we want to be able to
    document hooks even if their dependencies are not
    installed.

    :param name: Name of the hook
    :return: The documentation module for the hook.
    """
    file_path = Path(__file__).parent / name / "documentation.py"
    if not file_path.exists():
        raise FileNotFoundError(
            f"The documentation.py file for hook '{name}' was not found. "
            "Cannot load the hook's hyperparameter specification."
        )
    spec = importlib.util.spec_from_file_location(
        f"metatrain.utils.hooks.{name}.documentation", file_path
    )
    assert spec is not None  # for mypy
    documentation = importlib.util.module_from_spec(spec)
    assert spec.loader is not None  # for mypy
    spec.loader.exec_module(documentation)
    sys.modules[f"metatrain.utils.hooks.{name}.documentation"] = documentation
    return documentation


def get_hypers_class(name: str) -> type:
    """
    Returns the hypers classes for a given hook.

    :param name: Name of the hook.
    :return: The hypers class for the hook.
    """
    documentation = preload_documentation_module(name)
    return documentation.Hypers


def get_default_hypers(name: str, base_precision: Optional[int] = None) -> dict:
    """Returns the default hook hyperparameters.

    When *base_precision* is ``None`` (the default), the returned dict may
    contain ``"${base_precision}"`` interpolation strings in precision fields.
    These resolve automatically after ``OmegaConf.merge`` with a config that
    provides ``base_precision`` at the root level.

    When *base_precision* is an integer (16, 32, or 64), all interpolations are
    resolved before returning so callers get a plain dict of concrete values.

    :param name: Name of the hook
    :param base_precision: If given, resolve ``${base_precision}`` interpolations
        to this value before returning.
    :return: Default hyperparameters of the hook
    """

    hypers_class = get_hypers_class(name)

    defaults = init_with_defaults(hypers_class)

    if base_precision is not None:
        cfg = OmegaConf.create({"base_precision": base_precision, **defaults})
        container = OmegaConf.to_container(cfg, resolve=True)
        container.pop("base_precision")
        return container

    return defaults


def write_hypers_yaml(
    name: str, output_path: Path | str, include_name: bool = False
) -> None:
    """Write YAML file with defaults for a given hook.

    Given a hook name, this function imports the corresponding
    module, finds out what the hyperparameters are for the hook
    and its trainer, and generates a YAML file with the default
    hyperparameters.

    :param name: The hook to generate the files for.
    :param output_path: The path to write the YAML file to.
    :param include_name: If True, the YAML file will contain a top-level
        key with the hook name, mimicking the input that needs to be
        provided to use the hook.
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
    """Should be used when a hook is asked to produce an output
    but the hook is not able to find the layout of such output.

    This is because the function that sets up hooks will go through
    the other hooks and then retry this one. Maybe some other hook
    has requested an input that matches this output's name and then
    the layout will now be available.
    """


def setup_hooks(
    dataset_info: DatasetInfo,
    hooks_hypers: Optional[dict[str, dict[str, str] | str]] = None,
) -> tuple[
    list[torch.nn.Module],
    dict[str, TargetInfo],
]:
    """
    Setup the post-processing hooks to apply to the outputs of the model.

    :param dataset_info: The dataset information, which contains target
        information and also the hooks specification.
    :return: A tuple containing the list of post-processing hooks, and a
        dictionary with the outputs that the model should produce before the
        hooks are applied.
    """
    dataset_info = dataset_info.copy()
    model_outputs = dataset_info.targets.copy()

    all_hook_outputs = []
    post_hooks = []

    def _set_up(
        post_hooks_hypers: dict, dataset_info: DatasetInfo
    ) -> tuple[dict, list]:
        unavailable_out_hooks = {}
        unavailable_errors = []
        for hook_name, hook_hypers in post_hooks_hypers.items():
            hook_class = load_hook(hook_name)

            if isinstance(hook_hypers, str):
                san_hook_hypers: dict[str, dict[str, str] | str] = {
                    "outputs": hook_hypers,
                }
            else:
                san_hook_hypers = cast(dict[str, dict[str, str] | str], hook_hypers)

            # Get hook and add it to the list of hooks.
            try:
                hook = hook_class(san_hook_hypers, dataset_info)
            except UnavailableOutputError as e:
                unavailable_out_hooks[hook_name] = hook_hypers
                unavailable_errors.append(e)
                continue
            post_hooks.append(hook)

            # The model will not need to produce the outputs that this hook
            # takes care of, so remove them from the list of model outputs.
            hook_outputs = hook.supported_outputs()
            for output_name in hook_outputs:
                if output_name in all_hook_outputs:
                    raise ValueError(
                        f"Output '{output_name}' is already produced by another hook. "
                        "A given output can only be produced by one hook."
                    )
                model_outputs.pop(output_name, None)
            all_hook_outputs.extend(list(hook_outputs.keys()))

            # Add the inputs that the hook requests to the model outputs,
            # unless they are already present somewhere.
            requested = hook.requested_target_infos()
            extra_data_info = dataset_info.extra_data
            for req_name, req_info in requested.items():
                if req_name in model_outputs:
                    continue
                elif req_name in all_hook_outputs:
                    continue
                elif req_name in extra_data_info:
                    continue
                else:
                    model_outputs[req_name] = req_info

        return unavailable_out_hooks, unavailable_errors

    hooks_to_set_up = hooks_hypers
    while True:
        new_hooks_to_set_up, errors = _set_up(hooks_to_set_up, dataset_info)
        if len(new_hooks_to_set_up) == 0:
            break
        elif len(new_hooks_to_set_up) == len(hooks_to_set_up):
            raise ValueError(
                "Some hooks could not be set up because they requested outputs "
                "that are not present either in the targets, extra data or "
                "in other hooks' inputs. \nThe errors were the following:\n - "
                + "\n - ".join(str(e) for e in errors)
            )
        # We managed to set up some hooks in this iteration, do another iteration
        # to see if we can set up more hooks now that some outputs are available.
        hooks_to_set_up = new_hooks_to_set_up
        dataset_info.targets.update(model_outputs)

    return post_hooks, model_outputs


def restart_hooks(
    instantiated_hooks: list[HookInterface],
    dataset_info: DatasetInfo,
    new_hooks_hypers: dict[str, dict[str, str] | str],
) -> tuple[list[HookInterface], dict[str, TargetInfo], dict]:
    """
    Restart hooks with new dataset information.

    :param hooks: List of existing hooks to restart.
    :param dataset_info: New dataset information to use for restarting the hooks.
    :param new_hooks_hypers: The hook hypers passed for the hooks restart.
    :return: A tuple containing:
      - the list of restarted hooks,
      - a dictionary with the outputs that the model should produce before
        the hooks are applied,
      - the hook hyperparameters that will regenerate the returned hooks
        if passed to setup_hooks(). These should be stored in the checkpoint
        of the model.
    """
    if len(instantiated_hooks) > 0:
        raise ValueError("Restarting from previous hooks is not supported yet.")

    forward_hooks, model_outs = setup_hooks(dataset_info, new_hooks_hypers)

    return forward_hooks, model_outs, new_hooks_hypers
