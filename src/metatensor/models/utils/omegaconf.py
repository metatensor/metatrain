from pathlib import Path
from typing import Union

from omegaconf import DictConfig, ListConfig, OmegaConf


def file_format(_parent_):
    """Custom OmegaConf resolver to find the file format.

    File format is obtained based on the suffix of the ``read_from`` field in the same
    section."""
    return Path(_parent_["read_from"]).suffix


# Register custom resolvers
OmegaConf.register_new_resolver("file_format", file_format)


def _resolve_single_str(config):
    if isinstance(config, str):
        return OmegaConf.create({"read_from": config})
    else:
        return config


# BASE CONFIGURATIONS
CONF_SYSTEMS = OmegaConf.create(
    {
        "read_from": "${..read_from}",
        "file_format": "${file_format:}",
        "key": None,
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
    """Expands shorthand notations in a dataset configuration to their full formats.

    This function takes a dataset configuration, either as a string, DictConfig or a
    ListConfig, and expands it into a detailed configuration format. It processes
    systems, targets, and gradient sections, setting default values and inferring
    missing information. Unknown keys are ignored, allowing for flexibility.

    If the dataset configuration is either a :class:`str` or a
    :class:`omegaconf.DictConfig`

    The function performs the following steps for each c

    - Loads base configurations for systems, targets, and gradients from predefined
      YAML files.
    - Merges and interpolates the input configuration with the base configurations.
    - Expands shorthand notations like file paths or simple true/false settings to full
      dictionary systems.
    - Handles special cases, such as the mandatory nature of the 'energy' section for MD
      simulations and the mutual exclusivity of 'stress' and 'virial' sections.
    - Validates the final expanded configuration, particularly for gradient-related
      settings, to ensure consistency and prevent conflicts during training.

    :param conf: The dataset configuration, either as a file path string or a DictConfig
        object.
    :returns: The fully expanded dataset configuration.
    :raises ValueError: If both ``virial`` and ``stress`` sections are enabled in the
        'energy' target, as this is not permissible for training.
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

                # Add default gradients "energy" target section
                if target_key == "energy":
                    # For special case of the "energy" we add the section for force and
                    # stress gradient by default
                    target = OmegaConf.merge(CONF_ENERGY, target)
                else:
                    target = OmegaConf.merge(CONF_TARGET, target)

                if target["key"] is None:
                    target["key"] = target_key

                # Update DictConfig to allow for config node interpolation
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

                # If user sets the virial gradient and leaves the stress section
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
        if (
            actual_options_element["systems"]["length_unit"]
            != desired_options_element["systems"]["length_unit"]
        ):
            raise ValueError(
                "`length_unit`s are inconsistent between one of the dataset options."
                f" {actual_options_element['systems']['length_unit']!r} "
                "!= "
                f"{desired_options_element['systems']['length_unit']!r}."
            )

        for target in actual_options_element["targets"]:
            if target in desired_options_element["targets"]:
                if (
                    actual_options_element["targets"][target]["unit"]
                    != desired_options_element["targets"][target]["unit"]
                ):
                    raise ValueError(
                        f"Units of target {target!r} are inconsistent"
                        " between one of the dataset options. "
                        f"{actual_options_element['targets'][target]['unit']!r} "
                        f"!="
                        f" {desired_options_element['targets'][target]['unit']!r}."
                    )
            else:
                raise ValueError(
                    f"Target {target!r} is not present in one of the given dataset "
                    "options."
                )


def check_options_list(dataset_config: ListConfig) -> None:
    """Perform consistency checks within one dataset config.

    This is useful if the dataset config is made of several datasets.

    The function checks (1) if ``length_units`` in each system section is the same.
    If the names of the ``"targets"`` sections are the same between the elements of the
    list of datasets also (2) the units must be the same.

    :param dataset_config: A List of configuration to be checked. In the list contains
        only one element no checks are performed.
    """

    if len(dataset_config) == 1:
        return

    desired_config = dataset_config[0]
    # save unit for each target seaction for later comparison
    unit_dict = {k: v["unit"] for k, v in desired_config["targets"].items()}

    for actual_config in dataset_config[1:]:
        if (
            actual_config["systems"]["length_unit"]
            != desired_config["systems"]["length_unit"]
        ):
            raise ValueError(
                "`length_unit`s are inconsistent between one of the dataset options."
                f" {actual_config['systems']['length_unit']!r} "
                "!= "
                f"{desired_config['systems']['length_unit']!r}."
            )

        for target_key, target in actual_config["targets"].items():
            # If a target section name is not part of the saved units we add it for
            # later comparison. We do not have to start the loop again becuase this
            # target section name is not present in one of the datasets checked before.
            if target_key not in unit_dict.keys():
                unit_dict[target_key] = target["unit"]

            if unit_dict[target_key] != target["unit"]:
                raise ValueError(
                    f"Units of target section {target_key!r} are inconsistent. Found "
                    f"{target['unit']!r} and {unit_dict[target_key]!r}!"
                )
