import importlib
from pathlib import Path
from typing import Dict, List, Union

import torch
from omegaconf import Container, DictConfig, ListConfig, OmegaConf
from omegaconf.basecontainer import BaseContainer

from .. import RANDOM_SEED
from .devices import pick_devices


def file_format(_parent_: Container) -> str:
    """Custom OmegaConf resolver to find the file format.

    File format is obtained based on the suffix of the ``read_from`` field in the same
    section."""
    return Path(_parent_["read_from"]).suffix


def _get_architecture_capabilities(conf: BaseContainer) -> Dict[str, List[str]]:
    architecture_name = conf["architecture"]["name"]
    architecture = importlib.import_module(f"metatensor.models.{architecture_name}")
    return architecture.__ARCHITECTURE_CAPABILITIES__


def default_device(_root_: BaseContainer) -> str:
    """Custom OmegaConf resolver to find the default device of an architecture.

    Device is found using the :py:func:metatensor.models.utils.devices.pick_devices`
    function."""

    architecture_capabilities = _get_architecture_capabilities(_root_)
    desired_device = pick_devices(architecture_capabilities["supported_devices"])

    if len(desired_device) > 1:
        return "multi-cuda"
    else:
        return desired_device[0].type


def default_precision(_root_: BaseContainer) -> int:
    """Custom OmegaConf resolver to find the default precision of an architecture.

    File format is obtained based on the architecture name and its first entry in the
    ``supported_dtypes`` list."""

    architecture_capabilities = _get_architecture_capabilities(_root_)

    # desired `dtype` is the first entry
    default_dtype = architecture_capabilities["supported_dtypes"][0]

    # base_precision has to be a integere and not a torch dtype
    if default_dtype in [torch.float64, torch.double]:
        return 64
    elif default_dtype == torch.float32:
        return 32
    elif default_dtype == torch.float16:
        return 16
    else:
        raise ValueError(
            f"architectures `default_dtype` ({default_dtype}) refers to an unknown "
            "torch dtype. This should not happen."
        )


def default_random_seed() -> int:
    """Return session seed in the range [0, 2**32)."""
    return RANDOM_SEED


# Register custom resolvers
OmegaConf.register_new_resolver("file_format", file_format)
OmegaConf.register_new_resolver("default_device", default_device)
OmegaConf.register_new_resolver("default_precision", default_precision)
OmegaConf.register_new_resolver("default_random_seed", default_random_seed)


def _resolve_single_str(config: str) -> DictConfig:
    return OmegaConf.create({"read_from": config})


# BASE CONFIGURATIONS
CONF_SYSTEMS = OmegaConf.create(
    {
        "read_from": "${..read_from}",
        "file_format": "${file_format:}",
        "length_unit": None,
    }
)

CONF_TARGET_FIELDS = OmegaConf.create(
    {
        "quantity": "energy",
        "read_from": "${...systems.read_from}",
        "file_format": "${file_format:}",
        "key": None,
        "unit": None,
    }
)

CONF_GRADIENTS = OmegaConf.create({"forces": False, "stress": False, "virial": False})
CONF_GRADIENT = OmegaConf.create(
    {
        "read_from": "${..read_from}",
        "file_format": "${file_format:}",
        "key": None,
    }
)

KNWON_GRADIENTS = list(CONF_GRADIENTS.keys())

# merge configs to get default configs for energies and other targets
CONF_TARGET = OmegaConf.merge(CONF_TARGET_FIELDS, CONF_GRADIENTS)
CONF_ENERGY = CONF_TARGET.copy()
CONF_ENERGY["forces"] = CONF_GRADIENT.copy()
CONF_ENERGY["stress"] = CONF_GRADIENT.copy()


def expand_dataset_config(conf: Union[str, DictConfig, ListConfig]) -> ListConfig:
    """Expands shorthand notations in a dataset configuration to its full format.

    This function takes a dataset configuration, either as a :py:class:str,
    :py:class:`omegaconf.DictConfig` or a :py:class:`omegaconf.ListConfig`, and expands
    it into a detailed configuration format. It processes systems, targets, and gradient
    sections, setting default values and inferring missing information. Unknown keys are
    ignored, allowing for flexibility.

    If the dataset configuration is either a :class:`str` or a
    :class:`omegaconf.DictConfig`

    The function performs the following steps for each config

    - Merges and interpolates the input configuration with the base configurations.
    - Expands shorthand notations like file paths or simple true/false settings to full
      dictionary systems. This includes setting the units to the base units of
      ``"angstrom"`` and ``"eV"``.
    - Handles special cases, such as the mandatory nature of the "energy" section for MD
      simulations and the mutual exclusivity of 'stress' and 'virial' sections.
      Additionally the gradient sections for "forces" are enables by default.

    :param conf: The dataset configuration, either as a file path string or a DictConfig
        object.
    :raises ValueError: If both ``virial`` and ``stress`` sections are enabled in the
        "energy" target, as this is not permissible for training.
    :returns: List of datasets configurations. If ``conf`` was a :class:`str` or a
        :class:`omegaconf.DictConfig` the list contains only a single element.
    """
    # Expand str -> DictConfig
    if isinstance(conf, str):
        read_from = conf
        conf = OmegaConf.create(
            {"systems": read_from, "targets": {"energy": read_from}}
        )

    # Expand DictConfig -> ListConfig
    if isinstance(conf, DictConfig):
        conf = OmegaConf.create([conf])

    # Perform expansion per config inside the ListConfig
    for conf_element in conf:
        if hasattr(conf_element, "systems"):
            if type(conf_element["systems"]) is str:
                conf_element["systems"] = _resolve_single_str(conf_element["systems"])

            conf_element["systems"] = OmegaConf.merge(
                CONF_SYSTEMS, conf_element["systems"]
            )

        if hasattr(conf_element, "targets"):
            for target_key, target in conf_element["targets"].items():
                if type(target) is str:
                    target = _resolve_single_str(target)

                # for special case "energy" we enable sections for `forces` and `stress`
                # gradients by default
                if target_key == "energy":
                    target = OmegaConf.merge(CONF_ENERGY, target)
                else:
                    target = OmegaConf.merge(CONF_TARGET, target)

                if target["key"] is None:
                    target["key"] = target_key

                # update DictConfig to allow for config node interpolation
                conf_element["targets"][target_key] = target

                # merge and interpolate possibly present gradients with default gradient
                # config
                for gradient_key, gradient_conf in conf_element["targets"][
                    target_key
                ].items():
                    if gradient_key in KNWON_GRADIENTS:
                        if gradient_conf is True:
                            gradient_conf = CONF_GRADIENT.copy()
                        elif type(gradient_conf) is str:
                            gradient_conf = _resolve_single_str(gradient_conf)

                        if isinstance(gradient_conf, DictConfig):
                            gradient_conf = OmegaConf.merge(
                                CONF_GRADIENT, gradient_conf
                            )

                            if gradient_conf["key"] is None:
                                gradient_conf["key"] = gradient_key

                            conf_element["targets"][target_key][
                                gradient_key
                            ] = gradient_conf

                # If user sets the virial gradient and leaves the stress gradient
                # untouched, we disable the by default enabled stress gradient section.
                base_stress_gradient_conf = CONF_GRADIENT.copy()
                base_stress_gradient_conf["key"] = "stress"

                if (
                    target_key == "energy"
                    and conf_element["targets"][target_key]["virial"]
                    and conf_element["targets"][target_key]["stress"]
                    == base_stress_gradient_conf
                ):
                    conf_element["targets"][target_key]["stress"] = False

                if (
                    conf_element["targets"][target_key]["stress"]
                    and conf_element["targets"][target_key]["virial"]
                ):
                    raise ValueError(
                        f"Cannot perform training with respect to virials and stress "
                        f"as in section {target_key}. Set either `virials: off` or "
                        "`stress: off`."
                    )

    return conf


def check_units(
    actual_options: Union[DictConfig, ListConfig],
    desired_options: Union[DictConfig, ListConfig],
) -> None:
    """Perform consistency checks between two dataset configs.

    :param actual_options: The dataset options that you want to test.
    :param desired_options: The dataset options ``actual_options`` is tested against.

    :raises ValueError: If the length units are not consistent between
        the system in the dataset options.
    :raises ValueError: If a target is present only in desider_option and
        not in actual_option.
    :raises ValueError: If the unit of a target quantity is not consistent between
        the dataset option.
    """
    if type(actual_options) is DictConfig:
        actual_options = OmegaConf.create([actual_options])
    if type(desired_options) is DictConfig:
        desired_options = OmegaConf.create([desired_options])

    if len(actual_options) != len(desired_options):
        raise ValueError(
            f"Length of actual_options ({len(actual_options)}) and desired_options "
            f"({len(desired_options)}) is different!"
        )

    for actual_options_element, desired_options_element in zip(
        actual_options,
        desired_options,
    ):
        actual_length_unit = actual_options_element["systems"]["length_unit"]
        desired_length_unit = desired_options_element["systems"]["length_unit"]

        if actual_length_unit != desired_length_unit:
            raise ValueError(
                "`length_unit`s are inconsistent between one of the dataset options. "
                f"{actual_length_unit!r} != {desired_length_unit!r}."
            )

        for target in actual_options_element["targets"]:
            actual_unit = actual_options_element["targets"][target]["unit"]
            if target in desired_options_element["targets"]:
                desired_unit = desired_options_element["targets"][target]["unit"]
                if actual_unit != desired_unit:
                    raise ValueError(
                        f"Units of target {target!r} are inconsistent between one of "
                        f"the dataset options. {actual_unit!r} != {desired_unit!r}."
                    )
            else:
                raise ValueError(
                    f"Target {target!r} is not present in one of the given dataset "
                    "options."
                )


def check_options_list(dataset_config: ListConfig) -> None:
    """Perform consistency checks within one dataset config.

    This is useful if the dataset config is made of several datasets.

    - The function checks if ``length_units`` in each system section are known and the
       same.
    - For unknown quantities a warning is given.
    - If the names of the ``"targets"`` sections are the same between the elements of
       the list of datasets also the units must be the same.

    :param dataset_config: A List of configuration to be checked. In the list contains
        only one element no checks are performed.
    :raises ValueError: If for a known quantity the units are not known.
    """
    desired_config = dataset_config[0]
    # save unit for each target seaction for later comparison
    unit_dict = {k: v["unit"] for k, v in desired_config["targets"].items()}

    desired_length_unit = desired_config["systems"]["length_unit"]

    # loop over ALL configs because we have check units for all elements in
    # `dataset_config`
    for actual_config in dataset_config:
        # Perform consistency checks between config elements
        actual_length_unit = actual_config["systems"]["length_unit"]
        if actual_length_unit != desired_length_unit:
            raise ValueError(
                "`length_unit`s are inconsistent between one of the dataset options."
                f" {actual_length_unit!r} != {desired_length_unit!r}."
            )

        for target_key, target in actual_config["targets"].items():
            unit = target["unit"]

            # If a target section name is not part of the saved units we add it for
            # later comparison. We do not have to start the loop again because this
            # target section name is not present in one of the datasets checked before.
            if target_key not in unit_dict.keys():
                unit_dict[target_key] = unit

            if unit_dict[target_key] != unit:
                raise ValueError(
                    f"Units of target section {target_key!r} are inconsistent. Found "
                    f"{unit!r} and {unit_dict[target_key]!r}!"
                )
