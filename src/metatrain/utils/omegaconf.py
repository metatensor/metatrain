import json
from typing import Any, Union

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.basecontainer import BaseContainer

from .. import PACKAGE_ROOT, RANDOM_SEED
from .architectures import import_architecture
from .devices import pick_devices
from .jsonschema import validate


def _get_architecture_model(conf: BaseContainer) -> Any:
    architecture = import_architecture(conf["architecture"]["name"])
    return architecture.__model__


def default_device(_root_: BaseContainer) -> str:
    """Custom OmegaConf resolver to find the default device of an architecture.

    Device is found using the :py:func:metatrain.utils.devices.pick_devices`
    function."""

    Model = _get_architecture_model(_root_)
    desired_device = pick_devices(Model.__supported_devices__)

    if len(desired_device) > 1:
        return "multi-cuda"
    else:
        return desired_device[0].type


def default_precision(_root_: BaseContainer) -> int:
    """Custom OmegaConf resolver to find the default precision of an architecture.

    File format is obtained based on the architecture name and its first entry in the
    ``supported_dtypes`` list."""

    Model = _get_architecture_model(_root_)

    # desired `dtype` is the first entry
    default_dtype = Model.__supported_dtypes__[0]

    # `base_precision` in options has to be a integer and not a torch.dtype
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


def default_huber_loss_delta() -> float:
    """Return the default delta for the huber loss."""
    return 1.0


# Register custom resolvers
OmegaConf.register_new_resolver("default_device", default_device)
OmegaConf.register_new_resolver("default_precision", default_precision)
OmegaConf.register_new_resolver("default_random_seed", lambda: RANDOM_SEED)
OmegaConf.register_new_resolver("default_loss_type", lambda: "mse")
OmegaConf.register_new_resolver("default_loss_reduction", lambda: "mean")
OmegaConf.register_new_resolver("default_loss_sliding_factor", lambda: None)
OmegaConf.register_new_resolver("default_loss_weight", lambda: 1.0)


def _resolve_single_str(config: str) -> DictConfig:
    return OmegaConf.create({"read_from": config})


# Base options/configurations
BASE_OPTIONS = OmegaConf.create(
    {
        "device": "${default_device:}",
        "base_precision": "${default_precision:}",
        "seed": "${default_random_seed:}",
    }
)


CONF_SYSTEMS = OmegaConf.create(
    {
        "read_from": "${..read_from}",
        "reader": None,
        "length_unit": None,
    }
)

CONF_TARGET_FIELDS = OmegaConf.create(
    {
        "quantity": "energy",
        "read_from": "${...systems.read_from}",
        "reader": None,
        "key": None,
        "unit": None,
        "per_atom": False,
        "type": "scalar",
        "num_subtargets": 1,
    }
)

CONF_EXTRA_FIELDS = OmegaConf.create(
    {
        "quantity": "",
        "read_from": "${...systems.read_from}",
        "reader": None,
        "key": None,
        "unit": None,
        "per_atom": False,
        "type": "scalar",
        "num_subtargets": 1,
    }
)

CONF_GRADIENTS = OmegaConf.create({"forces": False, "stress": False, "virial": False})
CONF_GRADIENT = OmegaConf.create(
    {
        "read_from": "${..read_from}",
        "reader": None,
        "key": None,
    }
)

CONF_LOSS = OmegaConf.create(
    {
        "type": "${default_loss_type:}",
        "weight": "${default_loss_weight:}",
        "reduction": "${default_loss_reduction:}",
        "sliding_factor": "${default_loss_sliding_factor:}",
        "gradients": {},
    }
)

KNOWN_GRADIENTS = list(CONF_GRADIENTS.keys())

# Merge configs to get default configs for energies and other targets
CONF_TARGET = OmegaConf.merge(CONF_TARGET_FIELDS, CONF_GRADIENTS)
CONF_ENERGY = CONF_TARGET.copy()
CONF_ENERGY["forces"] = CONF_GRADIENT.copy()
CONF_ENERGY["stress"] = CONF_GRADIENT.copy()
CONF_EXTRA_DATA = CONF_EXTRA_FIELDS.copy()

# Schema with the dataset options
with open(PACKAGE_ROOT / "share/schema-dataset.json") as f:
    SCHEMA_DATASET = json.load(f)


def check_dataset_options(dataset_config: ListConfig) -> None:
    """Perform consistency checks within one dataset config.

    This is useful if the dataset config is made of several datasets.

    - The function checks if ``length_units`` in each system section are known and the
       same.
    - For unknown quantities a warning is given.
    - If the names of the ``"targets"`` sections are the same between the elements of
       the list of datasets also the units must be the same.
    - Two targets with the names `{target}` and `mtt::{target}` are not allowed.

    :param dataset_config: A List of configuration to be checked. In the list contains
        only one element no checks are performed.
    :raises ValueError: If the units are not consistent between the dataset options or
        if two different targets have the `{target}` and `mtt::{target}` names.
    """
    desired_config = dataset_config[0]

    if hasattr(desired_config, "targets"):
        # save unit for each target seaction for later comparison
        unit_dict = {k: v["unit"] for k, v in desired_config["targets"].items()}
    else:
        unit_dict = {}

    if hasattr(desired_config, "systems"):
        desired_length_unit = desired_config["systems"]["length_unit"]
    else:
        desired_length_unit = None

    if hasattr(desired_config, "extra_data"):
        # save unit for each extra_data section for later comparison
        for extra_data_key, extra_data in desired_config["extra_data"].items():
            unit_dict[extra_data_key] = extra_data["unit"]

    # loop over ALL configs because we have check units for all elements in
    # `dataset_config`
    for actual_config in dataset_config:
        if desired_length_unit:
            # Perform consistency checks between config elements
            actual_length_unit = actual_config["systems"]["length_unit"]
            if actual_length_unit != desired_length_unit:
                raise ValueError(
                    "`length_unit`s are inconsistent between one of the dataset "
                    f"options. {actual_length_unit!r} != {desired_length_unit!r}."
                )

        if hasattr(actual_config, "targets"):
            for target_key, target in actual_config["targets"].items():
                unit = target["unit"]

                # If a target section name is not part of the saved units we add it for
                # later comparison. We do not have to start the loop again because this
                # target section name is not present in one of the datasets checked
                # before.
                if target_key not in unit_dict.keys():
                    unit_dict[target_key] = unit

                if unit_dict[target_key] != unit:
                    raise ValueError(
                        f"Units of target section {target_key!r} are inconsistent. "
                        f"Found {unit!r} and {unit_dict[target_key]!r}!"
                    )

        # `target` and `mtt::target` are not allowed to be present at the same time
        if hasattr(actual_config, "targets"):
            for target_key in actual_config["targets"].keys():
                if f"mtt::{target_key}" in actual_config["targets"].keys():
                    raise ValueError(
                        f"Two targets with the names `{target_key}` and "
                        f"`mtt::{target_key}` are not allowed to be present "
                        "at the same time."
                    )

        if hasattr(actual_config, "extra_data"):
            for extra_data_key, extra_data in actual_config["extra_data"].items():
                unit = extra_data["unit"]

                # If a extra_data section name is not part of the saved units we add it
                # for later comparison. We do not have to start the loop again because
                # this extra_data section name is not present in one of the datasets
                # checked before.
                if extra_data_key not in unit_dict.keys():
                    unit_dict[extra_data_key] = unit

                if unit_dict[extra_data_key] != unit:
                    raise ValueError(
                        f"Units of extra_data section {extra_data_key!r} are "
                        "inconsistent. "
                        f"Found {unit!r} and {unit_dict[extra_data_key]!r}!"
                    )


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
        validate(instance=OmegaConf.to_container(conf_element), schema=SCHEMA_DATASET)
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
                    if gradient_key in KNOWN_GRADIENTS:
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

                            conf_element["targets"][target_key][gradient_key] = (
                                gradient_conf
                            )

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

        if hasattr(conf_element, "extra_data"):
            for extra_data_key, extra_data in conf_element["extra_data"].items():
                if type(extra_data) is str:
                    extra_data = _resolve_single_str(extra_data)

                extra_data = OmegaConf.merge(CONF_EXTRA_DATA, extra_data)

                if extra_data["key"] is None:
                    extra_data["key"] = extra_data_key

                # update DictConfig to allow for config node interpolation
                conf_element["extra_data"][extra_data_key] = extra_data

    check_dataset_options(conf)
    return conf


def expand_loss_config(conf: DictConfig) -> DictConfig:
    """Expand the loss configuration to a list of configurations.

    :param conf: The loss configuration to expand.
    :returns: A list of expanded loss configurations.
    """

    training_confs = conf["training_set"]

    if not isinstance(training_confs, ListConfig):
        training_confs = OmegaConf.create([training_confs])

    # initialize
    loss_dict: dict = {}
    conf_loss = CONF_LOSS.copy()
    OmegaConf.resolve(conf_loss)
    train_on_forces = False
    train_on_stress_or_virial = False

    # fill loss_dict with default values
    for tc in training_confs:
        for target_name, opts in tc["targets"].items():
            if target_name == "energy":
                f, s = _process_energy(loss_dict, opts, conf_loss)
                train_on_forces |= f
                train_on_stress_or_virial |= s
            else:
                loss_dict[target_name] = conf_loss.copy()

    train_hypers = conf["architecture"]["training"]
    if "loss" not in train_hypers:
        # Use default loss configuration
        train_hypers["loss"] = OmegaConf.create(loss_dict)
    else:
        # Expand str -> DictConfig
        if isinstance(train_hypers["loss"], str):
            # TODO: add test
            # the string must be the loss type, which is going to be used
            # for all targets
            for t in loss_dict.keys():
                loss_dict[t]["type"] = train_hypers["loss"]
                if train_hypers["loss"] == "huber":
                    loss_dict[t]["delta"] = default_huber_loss_delta()
            train_hypers["loss"] = OmegaConf.create(loss_dict)

        else:
            # Expand per-target str loss configurations
            for t in loss_dict.keys():
                if t in train_hypers["loss"]:
                    if isinstance(train_hypers["loss"][t], str):
                        train_hypers["loss"][t] = {"type": train_hypers["loss"][t]}
                        if train_hypers["loss"][t]["type"] == "huber":
                            train_hypers["loss"][t]["delta"] = (
                                default_huber_loss_delta()
                            )

            # Adapt the loss configuration to the internal structure
            if train_on_forces:
                _migrate_gradient_key(train_hypers["loss"], "forces", "positions")
            else:
                if "forces" in train_hypers["loss"]:
                    del train_hypers["loss"]["forces"]

            if train_on_stress_or_virial:
                for legacy in ["stress", "virial"]:
                    _migrate_gradient_key(train_hypers["loss"], legacy, "strain")
            else:
                if "stress" in train_hypers["loss"]:
                    del train_hypers["loss"]["stress"]
                if "virial" in train_hypers["loss"]:
                    del train_hypers["loss"]["virial"]

            # Add default delta for huber loss if not present
            for t in train_hypers["loss"].keys():
                if "type" in train_hypers["loss"][t]:
                    if train_hypers["loss"][t]["type"] == "huber":
                        if "delta" not in train_hypers["loss"][t]:
                            train_hypers["loss"][t]["delta"] = (
                                default_huber_loss_delta()
                            )
                if "gradients" in train_hypers["loss"][t]:
                    for grad_key in train_hypers["loss"][t]["gradients"].keys():
                        if "type" in train_hypers["loss"][t]["gradients"][grad_key]:
                            if (
                                train_hypers["loss"][t]["gradients"][grad_key]["type"]
                                == "huber"
                            ):
                                if (
                                    "delta"
                                    not in train_hypers["loss"][t]["gradients"][
                                        grad_key
                                    ]
                                ):
                                    train_hypers["loss"][t]["gradients"][grad_key][
                                        "delta"
                                    ] = default_huber_loss_delta()

            train_hypers["loss"] = OmegaConf.merge(loss_dict, train_hypers["loss"])

    conf["architecture"]["training"] = train_hypers
    return conf


def _migrate_gradient_key(loss_dict: dict, old_key: str, grad_key: str):
    """
    If `old_key` exists in `loss_dict`, move it under
    loss_dict['energy']['gradients'][grad_key], creating the necessary nested dicts
    along the way.
    """
    if old_key in loss_dict:
        if "energy" not in loss_dict:
            loss_dict["energy"] = {}
        if "gradients" not in loss_dict["energy"]:
            loss_dict["energy"]["gradients"] = {}
        loss_dict["energy"]["gradients"][grad_key] = loss_dict[old_key]
        del loss_dict[old_key]


def _process_energy(
    loss_dict: dict,
    opts: dict,
    template: dict,
) -> tuple[bool, bool]:
    """
    Ensure `loss_dict["energy"]` exists, reset its gradients, and add 'positions' /
    'strain' entries if requested by opts.
    Returns (added_forces, added_strain) bools.
    """
    if "energy" not in loss_dict:
        loss_dict["energy"] = template.copy()
    # start with an empty gradients dict each time
    loss_dict["energy"]["gradients"] = {}

    added_forces = False
    added_strain = False

    if opts.get("forces", False):
        loss_dict["energy"]["gradients"]["positions"] = template.copy()
        added_forces = True

    if opts.get("stress", False) or opts.get("virial", False):
        loss_dict["energy"]["gradients"]["strain"] = template.copy()
        added_strain = True

    return added_forces, added_strain


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
